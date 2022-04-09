# python notebook for Make Your Own Neural Network
# code for a 3-layer neural network, and code for learning the MNIST dataset
# (c) Tariq Rashid, 2016, GPLv2


#%%time
import re
import numpy
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.special
#INTERACTIVE PLOTS. CHANGE TO AUTO OR INLINE AND INDUCE A KERNEL RESET
#IF PROBLEMS ARISE ON YOUR SYSTEM.
#%matplotlib notebook 

#CONSTANTS, NOT NECESSARILY USED IN EVERY SINGLE NN INSTANTIATION
I_NODES = 10
H_NODES = 200
O_NODES = 1
LR_SIG = 0.1 
LR_RELU = 0.01
#THIS CAN BE CHANGED TO "data/train/ai.txt" TO TRAIN THE NN ON THE ENTIRE TRAINING SET
#HOWEVER, THE HUGELY INCREASED PROCESSING TIME SIMPLY DOES NOT JUSTIFY THE VERY SLIGHT
#IMPROVEMENT IN PERFORMANCE
TRAIN_FN = "data/train/ai_nano.txt" 
TEST_FN = "data/test/ai_nano_t.txt"
card_dictionary = {
    "T":10,  #TEN
    "J":11,  #JACK
    "Q":12,  #QUEEN
    "K":13,  #KING
    "A":14,  #ACE
    "d":0.1, #DIAMONDS
    "s":0.2, #SPADES
    "c":0.3, #CLUBS
    "h":0.4  #HEARTS
}


#THIS FUNCTION OPENS THE FILE HANDED TO IT (RAW DATA SET) AND EXTRACTS INFORMATION
#LINE BY LINE AND SPLITS IT AUTOMATICALLY INTO PLAYS, PLAYERS, HANDS, TABLE CARDS, 
#AND OUTCOMES (WIN/LOSS). THE EXTRACTED INFORMATION THEN REQUIRES SOME FURTHER PARSING (BELOW)
def load_training_set(filename):
    #LOOP THROUGH RAW DATASET AND SPLIT EACH LINE INTO AN ELEMENT AND THEN EACH ELEMENT INTO
    #ANOTHER SUBLIST OF ELEMENTS WHENEVER WE FIND A ':'. THEN, REMOVE THE LAST VALUE IF IT IS EMPTY
    with open(filename,'r') as file:
        x = file.read()
        x = x.split("\n")
        x = [i.split(":") for i in x]
        for i in x: 
            if i[-1]=='': x.pop()
    
    #EMPTY LISTS
    plays=[]
    h1=[]
    h2=[]
    hands=[]
    outcomes=[]
    bots=[]
    t1=[]
    table_cards=[]

    #LOOP THROUGH EACH SUBLIST, AND EXTRACT EACH ELEMENT INTO IT'S RESPECTIVE LIST, AFTER 
    #SPLITTING WHERE APPROPRIATE
    for i in x:
        p_=i[0]
        p_=p_.split("/")
        plays.append(p_)
        h_=i[1]
        h_=h_.split("|")
        h1.append(h_)
        o_=i[2]
        o_=o_.split("|")
        outcomes.append(o_)
        b_=i[3]
        b_=b_.split("|")
        bots.append(b_)

    #PLAYER HANDS AND TABLE CARDS ARE COMPLICATED BECAUSE THEY END UP IN THE SAME ELEMENT
    #BECAUSE THEIR SEPARATOR IS NOT THE ONE USED TO SPLIT EACH PLAYER UP, THUS
    #WE HAVE TO EXTRACT THEM FROM THE LIST OF PLAYERS' HANDS USING REGEX AND PLACE THEM
    #INTO THEIR RESPECTIVE LISTS. FIRST, TABLE CARDS, THEN HANDS. TABLE CARDS
    #ARE SPLIT INTO SUBLISTS, OF ELEMENTS: [FLOP, TURN, RIVER]
    for i in h1:
        for u in i:
            y = re.sub('^[^/]*/?','', u)
            y = y.split("/")
            t1.append(y)

    for i in h1:
        for u in i:
            h2.append(re.search('^([^/]*)', u).group())

    #AFTER EXTRACTING THEM, EACH ROUND OF CARDS HAS BEEN SPLIT INTO TWO ELEMENTS. WE THEN
    #COMBINE THEM INTO SINGLE ELEMENT, SO WE HAVE 
    #    [    [PLAYER1_HANDS_ROUND1, PLAYER2_HANDS_ROUND1],
    #         [PLAYER1_HANDS_ROUND2, PLAYER2_HANDS_ROUND2],
    #         [PLAYER1_HANDS_ROUND3, PLAYER2_HANDS_ROUND3]
    #    ETC                                              ... ]
    h_i=iter(h2)
    hands=[[i,next(h_i, '')] for i in h_i]
    #AS MENTIONED ABOVE, BECAUSE OF THE NATURE OF THE SPLIT, THE TABLE CARDS LIST CONTAINS
    #ONE EMPTY ELEMENT FOR EACH ACTUAL ELEMENT, SO WE MUST GO THROUGH
    #AND DELETE EVERY SECOND ELEMENT
    table_cards=t1[1::2]

    return plays, hands, table_cards, outcomes, bots #RETURN THE EXTRACTED VALUES


#THIS FUNCTION ACCEPTS A CARD AS IT IS IN THE RAW DATA SET AND CONVERTS IT
#TO A NUMERICAL VALUE ACCORDING TO THE CARD DICTIONARY ABOVE. 
#I.E. A "2s" (2 OF SPADES) BECOMES THE NUMBER 2.2. A "Th" THE NUMBER 10.4 AND SO ON
def parse_card(*args):
    for n in args:
        a=[n[x:x+2] for x in range(0, len(n), 2)] #COUPLES THE VALUES GIVEN INTO 2s (2,d,T,c -> 2d,Tc)
        for b in a:
            r=0 #FINAL VALUE OF THE CARDS
            for i in b:
                if isinstance(i,str):
                    try:
                        r+=float(i) #IF ITS A NUMBER, SIMPLY ADD IT
                    except (ValueError,TypeError):
                        try:
                            r+=card_dictionary.get(i) #IF ITS NOT A NUMBER, FIND ITS VALUE IN THE DICT, THEN ADD IT
                        except:
                            pass #COMPLETELY INVALID HANDS WILL SIMPLY RETURN A "0"
            yield r

#REFERS TO THE PREVIOUS FUNCTION. USED IF A LIST OF CARDS IS GIVEN INSTEAD.
def parse_cards(args,target_Important):
    xx=0
    for n in args:
        if target_Important and b[xx][0]==target: #YIELD ONLY THE CARDS FOR THE PLAYER OF INTEREST
            yield list(parse_card(n[0]))
        elif target_Important and b[xx][0]!=target:
            yield list(parse_card(n[1]))
        else:
            yield list(parse_card(*n))
        xx+=1

#EXTRACTS PLAY VALUES
def parse_plays(args):
    for n in args:
        a_list=[] #LIST CONTAINING PLAY FOR EACH STREET IN THE ROUND
        for i in n:
            if i=="f":        #EDGE CASE; INSTANT FOLD
                a_list.append(['-1'])
                break
            a=re.split('([crf])',i)                  #SPLIT ALONG CHECKS, RAISES, AND FOLDS
            a = [x.replace('c', '0') for x in a]     #REPLACE "C" WITH "0"
            a = [x.replace('f', '-1') for x in a]    #REPLACE "F" WITH "-1"
            a_2 = [x for x in a if x!='r' and x!=''] #REMOVE EMPTY STRINGS AND "R"
            a_list.append(a_2) #ADD TO LIST FOR CURRENT ROUND
        yield a_list


#UNDERSTANDING HOW THIS FUNCTION WORKS EXACTLY ISN'T VERY IMPORTANT, BUT IT ESSENTIALLY 
#USES THE VALUES EXTRACTED FROM THE RAW DATA AND ARRANGES IT IN AN APPROPRIATE MANNER
#SO THAT EACH BETTING STAGE IS DIVIDED UP, AND EACH PLAY MADE IN THAT BETTING STAGE
#IS DIVIDED INTO TURNS. SO IF WE HAVE A CHECK, A BET, THEN A FOLD, WE WOULD END UP WITH
#THREE SEPARATE LISTS: ONE FOR THE CHECK, ONE FOR THE BET, ONE FOR THE FOLD. THIS MEANS
#THE NN CONSIDERS EVERY SINGLE ACTION INDIVIDUALLY. AT THE END OF THIS FUNCTION,
#AN ARRAY CONTAINING THE DATA THAT SHOULD BE USED AS INPUTS TO THE NN AND AN ARRAY
#CONTAINING DATA TO BE USED AS THE TARGETS FOR EACH INDIVIDUAL INPUT IS RETURNED.
def compile_training_set(plays_array,bots_array,cards_array,tables_array,outcomes_array,target='Intermission_2pn_2017'):
    final_output=[] #ARRAY THAT WILL BE RETURNED AT THE END
    final_output2=[]#FINAL ARRAY USED TO TEST NN
    counter=0       #USED AS AN INDEX TO CALCULATE WHO GOES FIRST AMONG OTHER THINGS 
                    #A COUNTER IS USED AS IT IS SIMPLER THAN enumerate()
    for q in plays_array:
        pt=[0,0,0,0,0]
        #IF THE TARGET IS THE FIRST IN THE LIST OF BOTS, THEN HOME GOES FIRST,
        #ELSE, OPPONENT GOES FIRST (I.E. HOME DOES NOT GO FIRST)
        h_st=True if bots_array[counter][0]==target else False
        #SET OUTCOME FOR THIS ROUND OF PLAY
        outcome_=outcomes_array[counter][0] if bots_array[counter][0]==target else outcomes_array[counter][1]
        y=0 #THE STREET OF PLAY WE'RE ON (PRE-FLOP, POST-FLOP, ETC.)
        #THE CARDS ARRAYS AUTOMATICALLY GET PROPERLY FORMATTED EARLIER SO NO NEED FOR IF/ELSE
        cards_=cards_array[counter] 
        for n in q:
            kk=0 #TURN OF PLAY IN THIS PARTICULAR STREET
            m=0  #PREVIOUS PLAY
            for i in n:
                if i=='0': #DISTINGUISH BETWEEN A CALL AND A CHECK; CHECKS CAN ONLY HAPPEN AT THE START OF THE STREET
                    if (kk-1)==-1: pass
                    else:
                        n[kk]=n[kk-1] #ITS A CALL SO SET IT EQUAL TO THE PREVIOUS BET AMOUNT
                
                #0=HOME'S CARD #1
                #1=HOME'S CARD #2
                #2=OPPONENT'S PLAY
                #3=FLOP CARD #1
                #4=FLOP CARD #2
                #5=FLOP CARD #3
                #6=TURN CARD
                #7=RIVER CARD
                #8=HOME'S PREVIOUS PLAY
                #9=OUTCOME
                jj = [[] for x in range(10)]
                jj[9]=float(outcome_)   #OUTCOME
                jj[8]=0                 #YOUR PREVIOUS PLAY. DEFAULT TO CHECK
                try:                    #TABLE CARDS
                    if y==0:            #PRE FLOP
                        jj[3:8]=0,0,0,0,0
                    elif y==1:          #POST FLOP
                        jj[3:6]=tables_array[counter][:3]
                        pt[:3]=jj[3:6]
                    elif y==2:          #TURN
                        jj[3:6]=pt[:3]
                        jj[6]=tables_array[counter][3]
                        pt[3]=jj[6]
                    elif y==3:          #RIVER
                        jj[3:7]=pt[:4]
                        jj[7]=tables_array[counter][4]
                        pt[4]=jj[7]
                        
                    if jj[6]==[]: jj[6]=0 #FILL IN THE TABLE CARDS, 
                    if jj[7]==[]: jj[7]=0 #IN CASE THE ARRAY GIVEN ISNT LONG ENOUGH
                except IndexError:      #LIST OUT OF RANGE ERROR
                    jj[3:8]=0,0,0,0,0   #REVERT TO DEFAULT, AVOID HALT DUE TO ERROR
                jj[2]=0                 #OPPONENT'S PREVIOUS PLAY. DEFAULT TO CHECK
                jj[1]=cards_[1]         #HOME'S CARD #2 
                jj[0]=cards_[0]         #HOME'S CARD #1
                hh=0                    #DEFAULTS TO CHECK
                if kk==0 and h_st:      #IF HOME STARTS, THERE WERE NO PREVIOUS MOVES, JUST THE NEXT MOVE
                    try:
                        hh=float(n[kk]) #HOME'S NEXT MOVE
                    except: 
                        hh=0
                    final_output2.append(hh)
                    final_output.append(jj)
                elif kk==(len(n)-1):    #IF WE ARE AT THE END OF THE STREET, WE DON'T GET TO PLAY
                    pass
                elif h_st and kk%2:     #IF HOME STARTED, ACT EVERY SECOND TURN
                    jj[2]=float(i)
                    jj[8]=float(m) 
                    try:
                        hh=float(n[kk+1])       #HOME'S NEXT MOVE
                    except:
                        hh=0
                    final_output2.append(hh)
                    final_output.append(jj)
                elif (not h_st) and (not kk%2): #IF OPPONENT STARTED, ...
                    jj[2]=float(i)
                    jj[8]=float(m) 
                    try:
                        hh=float(n[kk+1])    #HOME'S NEXT MOVE
                    except:
                        hh=0
                    final_output2.append(hh)
                    final_output.append(jj)
                else:                   #CATCH-ALL ELSE STATEMENT
                    pass
                m=i                     #SET STORAGE AND INCREASE COUNTERS
                kk+=1
            y+=1
        counter+=1
    return final_output,final_output2


#THIS FUNCTION GOES THROUGH THE SET PROVIDED TO IT, AND FINDS THE LARGEST VALUE
#IN THAT SET, THEN DIVIDES ALL BETS SO THAT THEY ARE REPRESENTED AS A PORTION
#OF THAT LARGEST VALUE. THIS THEREFORE MAKES ALL BET AMOUNTS BE BETWEEN 0 AND 0.9999...9
#IT ALSO WORKS WITH THE TARGET SETS BY SETTING is_targets_set TO TRUE.
def normalise_training_set(input_set,is_targets_set=False):
    #IF WE ARE BEING GIVEN THE TARGETS SET. I.E., A 1D LIST OF BET VALUES
    if is_targets_set==True:
        #PREVENTS ANY VALUE FROM BEING 1 BY ADDING A VERY SMALL AMOUNT TO THE LARGEST
        #BET, WHICH IS WHAT ALL OTHER BETS ARE SCALED AGAINST. THEREFORE, NOT EVEN THE
        #LARGEST BET ITSELF WILL BE A 1. THIS IS GOOD BECAUSE NNS PERFORM BETTER WHEN
        #VALUES ARE NOT HARD 0s AND 1s. 0s HOWEVER ARE FINE IN THIS INSTANCE AS THEY
        #DENOTE AN ENTIRELY DIFFERENT ACTION FROM BETTING (>0)
        co_=max(input_set)+0.0000001 
        #GO THROUGH THE ENTIRE SET
        for n in range(len(input_set)):
            if input_set[n]==0.0: #IF IT IS A 0 (CHECK) LEAVE IT AS IS
                pass
            elif input_set[n]==-1: #IF IT IS A -1, SCALE IT TO -0.05 
                input_set[n]=-0.05 
                #-0.05 IS AN ARBITRARY VALUE THAT SEEMS TO WORK WELL,
                #AS -1 LED TO THE NN BECOMING TOO WEAK. A BETTER VALUE COULD
                #PROBABLY BE FOUND WITH FURTHER EXPERIMENTATION BUT THIS
                #VALUE WORKS VERY WELL ANYWAY, USUALLY WITHOUT THE NNS 
                #BECOMING OVER OR UNDERFITTED.
            else:
                input_set[n]=(input_set[n]/co_) #ANY OTHER VALUE IS A BET. SCALE IT ACCORDINGLY.
    else:
        #WE ARE DEALING WITH THE INPUTS LIST. LIST ELEMENT 2, 8, AND 9 CONTAIN BETS,
        #SO WE GO THROUGH THE ENTIRE INPUTS LIST AND FIND THE LARGEST BET FOR EACH
        #ELEMENT AND SAVE IT INTO ITS CORRESPONDING VARIABLE
        co_2,co_8,co_9=0,0,0
        for n in input_set:
            if n[2]>co_2: co_2=n[2]
            if n[8]>co_8: co_8=n[8]
            if n[9]>co_9: co_9=n[9]
        #A VERY SMALL AMOUNT IS ADDED TO EACH LARGEST BET VALUE TO PREVENT 1s, AS ABOVE
        co_2+=0.0000001
        co_8+=0.0000001
        co_9+=0.0000001
        #GO THROUGH THE ENTIRE INPUT SET. WHERE A BET VALUE EXISTS, SCALE IT ACCORDING
        #TO THE LARGEST BET VALUE IN THAT ELEMENT. IF THE ELEMENT IS A CARD INSTEAD,
        #SCALE IT BY 100. THIS HELPS THE NN NOT FOCUS COMPLETELY ON CARDS, WHICH
        #WOULD OTHERWISE HAVE VALUES FAR GREATER THAN THE BET VALUES AND THUS A LARGER BIAS.
        for n in input_set:
            n[0]=n[0]/100
            n[1]=n[1]/100
            n[2]=n[2]/co_2
            n[3]=n[3]/100
            n[4]=n[4]/100
            n[5]=n[5]/100
            n[6]=n[6]/100
            n[7]=n[7]/100
            n[8]=n[8]/co_8
            n[9]=n[9]/co_9

#OUR "HERO", THE BOT WE ARE INTERESTED IN AS ITS NAME APPEARS IN THE RAW DATA
target="Intermission_2pn_2017"

#EXTRACT THE VALUES FROM THE TRAINING SET AND COMPILE THEM INTO ONE LIST.
#tra_[0] WOULD INCLUDE THE INPUTS FOR THE NN, AND tra_[1] WOULD INCLUDE
#THE OUTPUT DESIRED FOR EACH INPUT (I.E. HOW MUCH SHOULD BE BET)
p,h,t,o,b=load_training_set(TRAIN_FN)
tra_=compile_training_set(list(parse_plays(p)),
                          b,
                          list(parse_cards(h,True)),
                          list(parse_cards(t,False)),
                          o,
                          target)


#STANDARD NN CLASS AS I USED IN PREVIOUS SUBMISSIONS. THE NAME WAS MAINTAINED JUST
#FOR THE SAKE OF MAKING THE CODE EASIER TO ADAPT, BUT IT IS NOT JUST A SIGMOID NN,
#AS ANY ONE OF SIG, RELU, OR SWISH CAN BE USED DEPENDING ON THE VALUE PASSED TO class_id.
#IT IS PRACTICALLY THE STANDARD CODE WITH SOME VERY SLIGHT MODIFICATIONS.
class nn_sig:
    #STANDARD INITIATION FUNCTION WITH SLIGHT CHANGE TO HOW WEIGHTS ARE INITIALISED
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, class_id):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.name=class_id
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        #THIS IS THE ONLY MODIFICATION; IT SIMPLY SCALES THE WEIGHTS IN BOTH INSTANCES.
        #THIS IS (A FORM OF) XAVIER INITIALISATION AND HELPS PREVENT THE ISSUE OF DIMINSIHING
        #GRADIENTS WITH THE SIGMOID ACTIVATION FUNCTION. I ALSO FOUND THIS PRODUCED
        #CLOSER VALUES ON AVERAGE IN BETWEEN SEVERAL DIFFERENT RUNS OF THE SAME CODE.
        self.wih=math.sqrt(1./self.inodes)*self.wih 
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.who=math.sqrt(1./self.hnodes)*self.who
        self.lr = learningrate
    
    #ACTIVATION FUNCTION. RETURNS A DIFFERENT VALUE BASED ON WHAT
    #ACTIVATION FUNCTION TYPE THIS NN IS
    def activation_function(self, x):
        if self.name=="relu":
            y = x*(x > 0)
            return y
        elif self.name=="sig":
            return scipy.special.expit(x)
        elif self.name=="swish": #SWISH ACTIVATION FUNCTION AND DERIVATIVE WERE IMPLEMENTED
            return (x*scipy.special.expit(x))
    
    #DERIVATIVE FUNCTION. RETURNS A DIFFERENT DERIVATIVE BASED ON WHAT
    #ACTIVATION FUNCTION TYPE THIS NN IS
    def deriv(self, x):
        if self.name=="relu":
            y =(x >0) * 1.0
            return y
        elif self.name=="sig":
            return x*(1.0-x)
        elif self.name=="swish":
            return self.activation_function(x)+(scipy.special.expit(x)*(1-self.activation_function(x)))
    
    #STOCK STANDARD TRAIN FUNCTON
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2)
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        self.who += self.lr * numpy.dot((output_errors * self.deriv(final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * self.deriv(hidden_outputs)),
                                        numpy.transpose(inputs)) 

    #STOCK STANDARD QUERY FUNCTON
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    
    #THESE FUNCTIONS CAN SAVE AND LOAD THE WEIGHT VALUES OF THE NN AND
    #ARE VERY HELPFUL WHEN I WANTED TO EXACTLY REPLICATE A SURFACE MAP OR
    #RESULT THAT I FORGOT TO RECORD OR DIDN'T GET ENOUGH SCREENSHOTS OF.
    #THIS WAS ESSENTIALLY DEPRECATED FOR THE USE OF PLOTS HOWEVER, AS
    #CODE TO SAVE THE PLOTS THEMSELVES (BYPASSING ANY NEED TO SAVE THE NN)
    #WAS LATER IMPLEMENTED (BELOW)
    def save_weights(self):
        with open("data/who.txt",'w') as file:
            print(self.who)
            numpy.savetxt(file, self.who)
        with open("data/wih.txt", 'w') as file:
            print(self.wih)
            numpy.savetxt(file, self.wih)
    def load_weights(self):
        with open("data/who.txt",'r') as file:
            self.who = numpy.loadtxt(file)
            print(self.who)
        with open("data/wih.txt",'r') as file:
            self.wih = numpy.loadtxt(file)
            print(self.wih)
    

    
#THIS FUNCTION QUERIES THE NETWORK AND COMPARES ITS OUTPUTS TO THE TEST SET
#IT WILL THEN RETURN ITS SCORECARD AND PERFORMANCE VALUE DERIVED FROM THAT
#SCORECARD. IT CAN GENERATE ONE OF TWO PERFORMANCE VALUES: THAT WHICH TAKES
#INTO CONSIDERATION CHECKS AND BETS BEYOND THAT OF THE TEST SET, AND THAT
#WHICH DOES NOT. THE ONE THAT DOES NOT IS A HARSHER MARKER, AS A NN CANNOT
#ACHIEVE A HIGH PERFORMANCE SIMPLY BY OUTPUTTING "0" AT EVERY CHANCE (CHECKING
#ALWAYS), AND IS A FAIR ALTERATION, AS BETTING WHERE ONE WOULD CHECK, AND
#BETTING HIGHER THAN THE TEST SET WOULD RESULT IN THE SAME OUTCOME AS 
#WHATEVER OTHER VALUE THE TEST SET PROPOSES. I.E. IF ONE'S ONLY TWO OPTIONS
#ARE TO FOLD (THEREBY FORFEITING ALL THAT HAS BEEN BET AT THAT ROUND AND
#EFFECTIVELY LOSING THAT ROUND) OR CHECK, ONE WOULD OBVIOUSLY CHECK AS IT
#WOULD CAUSE NO FINANCIAL LOSS (AS A CHECK DOES NOT REQUIRE BETTING) AND
#WOULD BE THE BEST CHOICE AS IT WOULD INCREASE THE CHANCES OF WINNING THE ROUND.
#THE SAME CAN BE SAID FOR LARGER THAN REQUIRED BETS, AS THE OPPONENT WOULD
#HAVE FOLDED REGARDLESS, IT WILL ALSO CAUSE OPPONENTS TO RECONSIDER THEIR
#BLUFFS, AND WOULD PUT MORE MONEY INTO PLAY (AS THE OPPONENT MUST MATCH
#OUT BET), THEREBY LEADING US TO BE MORE PROFITABLE.
def nn_test(n,testing_array,targets_array,return_average=True,consider_checks=False):
    
    scorecard = []
    
    for x in zip(testing_array,targets_array):
        
        outputs = n.query(x[0])
        
        #TARGET IS 0, SO A CHECK, SO IT DOES NOT MATTER HOW MUCH THE NETWORK CHOSE TO BET, IN THIS INSTANCE.
        #SEE ABOVE FOR A FULL EXPLANATION.
        if x[1]==0:
            scorecard.append(1)
            
        #BET AMOUNT IS LARGER THAN REQUIRED. THIS IS ALSO CONSIDERED A FULFILLED TARGET.
        #SEE ABOVE FOR A FULL EXPLANATION.
        elif outputs[0]>x[1]: 
            scorecard.append(1)
            
        else: #ALL OTHER SITUATIONS; EVALUATE SCORE AS A PROPORTION OF TARGET
            scorecard.append(abs(outputs[0]/x[1]))
            
    if return_average==True: #RETURN THE PERFORMANCE VALUE ACROSS THE ENTIRE SCORECARD
        #"HARSH" MARKING OF PERFORMANCE (SEE ABOVE)
        if consider_checks==False:
            temp_score=[x for x in scorecard if x!=1]
            scorecard_array = numpy.asarray(temp_score)
        elif consider_checks==True: #LESS HARSH MARKING
            scorecard_array = numpy.asarray(scorecard)
        else: #NONSENSE VALUES; RETURN BOTH JUST TO BE SAFE. NEVER USED UNLESS A MISTAKE IS MADE
            temp_score=[x for x in scorecard if x!=1]
            scorecard_array = numpy.asarray(temp_score)
            scorecard_array2 = numpy.asarray(scorecard)
            return scorecard,(scorecard_array.sum()*100/scorecard_array.size,scorecard_array2.sum()*100/scorecard_array2.size)
        return scorecard, (scorecard_array.sum()*100/scorecard_array.size)
    else: #DO NOT RETURN THE PERFORMANCE VALUE, ONLY THE SCORECARD
        return scorecard


#EXTRACT THE VALUES FROM THE TEST SET USING THE SAME FUNCTION
#AS WAS USED FOR THE TRAINING SET
p_t,h_t,t_t,o_t,b_t=load_training_set(TEST_FN)

#COMPILE THE TEST SET TOGETHER FROM ALL THE VALUES EXTRACTED
#FROM THE TEST SET
tes_=compile_training_set(list(parse_plays(p_t)),
                          b_t,
                          list(parse_cards(h_t,True)),
                          list(parse_cards(t_t,False)),
                          o_t,
                          target)

#NORMALISE THE TRAINING AND TEST SETS
normalise_training_set(tra_[0])
normalise_training_set(tra_[1],True)
normalise_training_set(tes_[0])
normalise_training_set(tes_[1],True)


#THIS FUNCTION GENERATES PLOTS DEPENDING ON THE plot_num AND HYPERPARAMETERS PROVIDED.
#SEE EACH OPTION FOR FURTHER INFORMATION
def generate_plots(plot_num,epoch_num=100,starting_lr=0.0,no_of_nns_=20,hidden_nodes=10):
    if plot_num==1:
        #THIS OPTION CREATES A 2D LIST OF NNS, EACH WITH DIFFERENT LEARNING RATES AND NO. OF HIDDEN NODES,
        #THEN TRAINS THEM ALL FOR THE NUMBER OF EPOCHS PROVIDED, AND PLOTS ALL PERFORMANCE VALUES
        #AS A SURFACE MAP
        size_=no_of_nns_+1
        final_list=[]
        #CREATE A 2D LIST OF NNS WITH DIFFERENT HYPERPARAMETERS
        nn_list = [[nn_sig(I_NODES,x,O_NODES,(starting_lr+(i/size_)),"swish") for i in range(size_)] for x in range(1,hidden_nodes+1)]
        
        for x in nn_list:
            pre_list=[]
            for x_ in x:
                for n in range(epoch_num):
                    for y in zip(tra_[0],tra_[1]): #TRAINS EACH NN
                        x_.train(y[0],y[1])
                pre_list.append(nn_test(x_,tes_[0],tes_[1],True,False)[1]) #TESTS EACH NN
            final_list.append(pre_list)
        
        #PLOTS EACH NN'S PERFORMANCE ON A SURFACE MAP
        x = numpy.linspace(starting_lr,1+starting_lr,no_of_nns_+1)
        y = range(0,len(nn_list))
        xs, ys = numpy.meshgrid(x, y)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Hidden Nodes')
        ax.set_zlabel('Performance')
        ax.set_title("Neural Network Performance")
        ax.plot_surface(xs, ys, numpy.array(final_list), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        
        #SAVES OUTPUTS IF REQUIRED AT ANY FURTHER DATE
        with open("plot1.txt", 'w') as file:
            numpy.savetxt(file, final_list)
    elif plot_num==2:
        #THIS OPTION TRAINS A NN (THE ACTIVATION FUNCTION MUST BE HAND CODED IN AT EACH INSTANCE)
        #FOR THE NUMBER OF EPOCHS GIVEN, AND INCREASES ITS LEARNING RATE AFTER EACH COMPLETE
        #TRAINING FOR THE NUMBER OF EPOCHS. THEN IT PLOTS THE DATA AS A SURFACE MAP
        final_list=[]
        epochs=epoch_num
        no_of_nns=no_of_nns_
        lr_=starting_lr
        for c in range(no_of_nns+1): #NUMBER OF NETWORKS BECOMES THE NUMBER OF INCREASES TO LEARNING RATE
            temp_list=[]
            #THE ACTIVATION FUNCTION MUST BE CHANGED BY HAND IF ONE DESIRES FOR E.G. RELU
            #THIS MERELY SIMPLIFIES THE CODE
            x1_=nn_sig(I_NODES,hidden_nodes,O_NODES,lr_,"swish") 
            for n in range(epochs): #NUMBER OF EPOCHS
                
                for x in zip(tra_[0],tra_[1]): #TRAIN THE NN
                    x1_.train(x[0],x[1])
                temp_list.append(nn_test(x1_,tes_[0],tes_[1],True,False)[1]) #APPEND ITS OUTPUT TO A LIST
                
            final_list.append(temp_list) 
            lr_+=(1/no_of_nns) #INCREASE THE LEARNING RATE
        
        #PLOTTING
        x = range(0,epoch_num)
        y = numpy.linspace(starting_lr,1+starting_lr,no_of_nns+1)
        xs, ys = numpy.meshgrid(x, y)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Learning Rate')
        ax.set_zlabel('Performance')
        ax.set_title("Neural Network Perf.")
        ax.plot_surface(xs, ys, numpy.array(final_list), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        
        #SAVES OUTPUTS
        with open("plot2.txt", 'w') as file:
            numpy.savetxt(file, final_list)
    elif plot_num==3:
        #TRAINS ALL NNS FOR THE NUMBER OF EPOCHS PROVIDED, AT THE HYPERPARAMETERS
        #PROVIDED, THEN TESTS THEM ALL AND PLOTS THE OUTPUT AT EACH EPOCH
        n1 = nn_sig(I_NODES,hidden_nodes,O_NODES,starting_lr,"swish")
        n2 = nn_sig(I_NODES,hidden_nodes,O_NODES,starting_lr,"sig") #INITIALISE NNS
        n3 = nn_sig(I_NODES,hidden_nodes,O_NODES,starting_lr,"relu")
        zzpp=[]
        zzpp2=[] #EMPTY LISTS THAT THE OUTPUT WILL BE APPENDED TO
        zzpp3=[]

        for n in range(epoch_num): #TRAINS ALL THREE NNS FOR THE NUMBER OF EPOCHS PROVIDED
            for x in zip(tra_[0],tra_[1]):
                n1.train(x[0],x[1])
                n2.train(x[0],x[1])
                n3.train(x[0],x[1])
            zzpp.append(nn_test(n1,tes_[0],tes_[1],True,False)[1])
            zzpp2.append(nn_test(n2,tes_[0],tes_[1],True,False)[1]) #APPENDS THEIR OUTPUTS TO THE LISTS
            zzpp3.append(nn_test(n3,tes_[0],tes_[1],True,False)[1])
        
        #PLOTTING
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(111)
        
        fig.suptitle('Activation Functions Effect on Performance')
        plt.xlabel('Epochs')
        plt.ylabel('Performance (%)')
        
        plt.plot(zzpp,color='blue', alpha=0.8, label='Swish')
        plt.plot(zzpp2,color='red',alpha=0.8, label='Sigmoid')
        plt.plot(zzpp3,color='green',alpha=0.8,label='ReLU')
        plt.legend()
        
        print(max(zzpp),"swish",numpy.argmax(zzpp)+1)
        print(max(zzpp2),"sig",numpy.argmax(zzpp2)+1) #OUTPUTS THE HIGHEST PERFORMANCE VALUE AND
        print(max(zzpp3),"relu",numpy.argmax(zzpp3)+1)#THE EPOCH AT WHICH IT WAS RECORDED, FOR EACH NN
        
        plt.plot(numpy.argmax(zzpp),max(zzpp),'bo')
        plt.plot(numpy.argmax(zzpp2),max(zzpp2),'ro') #PLOTS A DATA POINT AT THE HIGHEST PERFORMANCE VALUE
        plt.plot(numpy.argmax(zzpp3),max(zzpp3),'go')
        
    elif plot_num==4: 
        #TRAINS ALL THREE FUNCS AT THEIR OPTIMAL HYPERPARAMETERS, THEN TESTS THEM ON THE
        #TEST SET, AND PLOTS THEIR OUTPUT FOR EACH TEST VALUE, AS WELL AS THE TEST VALUE
        #ITSELF. THIS ALLOWS ONE TO SEE HOW THE NN BETS. A PERFORMANCE VALUE IS ALSO 
        #CALCULATED FOR EACH ACTIVATION FUNCTION
        n1 = nn_sig(I_NODES,1000,O_NODES,0.1,"swish")
        n2 = nn_sig(I_NODES,100,O_NODES,0.01,"sig") #INITIALISE THE NNS WITH OPTIMAL HYPERPARAMETERS
        n3 = nn_sig(I_NODES,10,O_NODES,0.01,"relu")
        zzpp=[]
        zzpp2=[] #EMPTY LISTS THAT THE OUTPUTS WILL BE APPENDED TO
        zzpp3=[]
        
        for n in range(606): #OPTIMAL NUMBER OF EPOCHS
            for x in zip(tra_[0],tra_[1]):
                n1.train(x[0],x[1])
        for n in range(13):  #OPTIMAL NUMBER OF EPOCHS
            for x in zip(tra_[0],tra_[1]):
                n2.train(x[0],x[1])
        for n in range(37):  #OPTIMAL NUMBER OF EPOCHS
            for x in zip(tra_[0],tra_[1]):
                n3.train(x[0],x[1])
        
        zzpp=n1.query(tes_[0])
        zzpp2=n2.query(tes_[0]) #TEST NNS
        zzpp3=n3.query(tes_[0])
        
        #PLOTTING
        fig = plt.figure(figsize=(14, 7))
        
        fig.suptitle('Activation Functions Effect on Performance')
        ax1 = plt.subplot('111')
        ax1.set_xlabel('Turn of Play')
        ax1.set_ylabel('Output (Bet Amount as Portion of Largest Single Bet)')
        
        ax1.plot(tes_[1],color='black', alpha=0.9, label='Test Set')
        ax1.plot(zzpp[0],color='b',alpha=0.6, label='Swish')
        ax1.plot(zzpp2[0],color='r',alpha=0.5, label='Sigmoid')
        ax1.plot(zzpp3[0],color='g',alpha=0.5, label='ReLU')
        ax1.legend()
        
        #SCALE THE Y MAX BY e/2 MULTIPLIED BY THE LARGEST OUTPUT OF SWISH. THIS ALLOWS
        #ALL OUTPUTS VALUES TO BE APPRECIATED; NONE IS TOO SMALL TO BE SEEN AND THOSE
        #THAT ARE TOO LARGE CAN STILL BE APPRECIATED
        ax1.set_ylim(min(zzpp[0]),max(zzpp[0])*(numpy.e/2))
        print(nn_test(n1,tes_[0],tes_[1])[1], "/", nn_test(n1,tes_[0],tes_[1],True,True)[1]," Swish")
        print(nn_test(n2,tes_[0],tes_[1])[1], "/", nn_test(n2,tes_[0],tes_[1],True,True)[1]," Sig")
        print(nn_test(n3,tes_[0],tes_[1])[1], "/", nn_test(n3,tes_[0],tes_[1],True,True)[1]," ReLU")
        ax1.set_xlim(-2,None)
        
        #SAVES OUTPUTS
        with open("plot4.1.txt", 'w') as file:
            numpy.savetxt(file, zzpp)
        with open("plot4.2.txt", 'w') as file:
            numpy.savetxt(file, zzpp2)
        with open("plot4.3.txt", 'w') as file:
            numpy.savetxt(file, zzpp3)
        
    #REGARDLESS OF THE PLOT GENERATED, SAVE AN IMAGE OF IT IN THE CURRENT DIRECTORY
    fig.savefig('plot.jpg')
    #SHOW THE PLOT ON SCREEN
    plt.show()

#UNCOMMENT ANY OF THESE AND RUN THIS CELL TO GENERATE PLOTS 
#FEEL FREE TO CHANGE ANY OF THE VALUES IF YOU'D LIKE
#generate_plots(1,100,0.0,20,10)
#generate_plots(1)
#generate_plots(2)
#generate_plots(3)
#generate_plots(4)
