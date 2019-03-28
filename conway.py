#!/usr/bin/env python

''' 
conway.py: For solving generalized Penney's game with 
generalized Conway formula, including simulations.

For background, see Miller(2019) '' 
'''

import numpy as np


__author__ = "Joshua B. Miller"
__copyright__ = "Creative Commons"
__credits__ = "none"
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Joshua B. Miller"
__email__ = "joshua.benjamin.miller@gmail.com"
__status__ = "Prototype"


def payoff_to_B_bets_if_A_occurs_first(A,B,alphabet):
    ''' (string, string, dictionary)-> (float)
    The fair payoff to all B bets if pattern A appears first.
    This function calculates the fair payoff to someone who initiates a 
    fresh sequence of bets each period in which bets anticipate pattern B

    note: Assuming the sequence ends at trial t>len(B), then when pattern A occurs
    there will be up to len(B) ongoing, and overlapping, B-bet sequences  .

    For example:
    >>>A='THH'
    >>>B='HHH'
    >>>alphabet={'T':.5, 'H':.5})
    >>>AB=payoff_to_B_bets_if_A_occurs_first(A,B,alphabet)
    Then in this case AB=4+2=6 as B betters who enter at T-2 lose immediately,
    those who enter at T-1 win twice,   and those who enter at T win once. 


    '''
    #make sure alphabet is a valid categorical distribution
    #(tolerate 1e-10 deviation; not too strict on the sum for precision issues)
    if abs(sum(alphabet.values())-1) > 1e-10:
        raise Exception("Alphabet is not a valid probability distribution")

    #make sure keys are strings
    if any( type(el) is not str for el in alphabet.keys() ) :
        raise Exception("only strings please")
    
    #make sure strings are of length 1
    if any( len(el)>1 for el in alphabet.keys() ) :
        raise Exception("Strings must be length 1")


    #Make sure all characters in the patterns appear in the Alphabet
    if any(char not in alphabet.keys() for char in A+B  ):
        raise Exception("All chacters must appear in the Alphabet")

    #make sure B is not a strict substring of A (or it will appear first for sure) 
    # and vice-versa
    if ( len(B)<len(A) and A.find(B)>-1) or ( len(A)<len(B) and B.find(A)>-1):
        raise Exception("one string cannot be a strict substring of another")
    
    # Calculate AB, the total payoffs from each sequence of bets anticipating pattern B
    # that are still active when the sequence stops at A
    AB = 0
    for i in range(len(A)):
        A_trailing = A[i:]
        B_leading = B[0:len(A_trailing)]
        if A_trailing == B_leading:
            #The sequence of bets anticipating B that are initiated at i (relatively)
            #need to be paid when A occurs if there is perfect overlap of the leading characters of B 
            # with the trailing characters of A
            #Why?The person waiting for B to occcur hasn't gone bankrupt yet, 
            #This person gets paid for betting correctly on every realization in A_trailing

            #On bet i, "wealth" is the amount invested predicting the event A_trailing[i],
            #this investment gets a fair gross rate of return
            #equal to the inverse of the probability of the event (1/alphabet[A_trailing[i]])
            wealth=1
            for i in range(len(A_trailing)):
                gross_return = 1/alphabet[A_trailing[i]]
                wealth = wealth*gross_return
            AB = AB + wealth
    return AB

def oddsAB(A,B,alphabet):
    ''' (string, string, dictionary)-> [list]
    returns odds against pattern A preceding pattern B
    odds[0] = "chances against A"
    odds[1] = "chances in favor of A"
    
    note: odds= 2* Conway's odds; see Miller (2019) for proof
    '''
 
    if A==B:
        raise Exception("A==B; patterns cannot precede themselves")
    elif ( len(B)<len(A) and A.find(B)>-1): #if B is strict substring of A
        odds = [1,0]    
    elif ( len(A)<len(B) and B.find(A)>-1): #if A is strict substring of B
        odds = [0,1] 
    else:
        AA=  payoff_to_B_bets_if_A_occurs_first(A,A,alphabet)
        AB = payoff_to_B_bets_if_A_occurs_first(A,B,alphabet)
        BB = payoff_to_B_bets_if_A_occurs_first(B,B,alphabet)
        BA = payoff_to_B_bets_if_A_occurs_first(B,A,alphabet)
        odds = [AA-AB , BB-BA]

    return odds

def probAB(A,B,alphabet):
    ''' (string, string, dictionary)-> (float)
            probability pattern A precedes pattern B
            note: odds are o[0] chances against for every o[1] chances in favor
            there are o[0]+o[1]
    '''
    o = oddsAB(A,B,alphabet)
    return o[1]/(o[0]+o[1])

def expected_waiting_time(A,B,alphabet):
    ''' (string, string, dictionary)-> (float)
            expected waiting time until the first occurance of A or B
            see Miller (2019) for derivation
    '''

    if A==B:
        wait = payoff_to_B_bets_if_A_occurs_first(A,A,alphabet)
    elif ( len(B)<len(A) and A.find(B)>-1): #if B is strict substring of A
        wait = payoff_to_B_bets_if_A_occurs_first(B,B,alphabet) 
    elif ( len(A)<len(B) and B.find(A)>-1): #if A is strict substring of B
        wait = payoff_to_B_bets_if_A_occurs_first(A,A,alphabet)
    else:
        AA=  payoff_to_B_bets_if_A_occurs_first(A,A,alphabet)
        AB = payoff_to_B_bets_if_A_occurs_first(A,B,alphabet)
        BB = payoff_to_B_bets_if_A_occurs_first(B,B,alphabet)
        BA = payoff_to_B_bets_if_A_occurs_first(B,A,alphabet)
        wait = (AA*BB - AB*BA)/(AA + BB - AB - BA)
    return wait

def simulate_winrates_penney_game(A,B,alphabet,number_of_sequences):
    '''
    (string, string, dictionary, integer)-> (list)
    Play generalized Penney's game and calculate how often
    pattern A precedes pattern B, and vice versa

    '''
    N = number_of_sequences
    
    #The letters in the dicitonary have a categorical distribution
    #defined by the key, value pairs
    outcomes = list(alphabet.keys())
    probabilities = list(alphabet.values())
    
    n_wins = np.array([0, 0])
    n_flips = 0
    
    for i in range(N):
        max_length=max(len(A),len(B))
        window = ['!']* max_length
        #on each experiment draw from dictionary until either pattern A,
        # or pattern B appears
        while True:
            window.pop(0)
            draw=np.random.choice(outcomes, 1, replace=True, p=probabilities)
            n_flips += 1
            window.append(draw[0])
            ch_window = "".join(map(str,window))
            if ch_window[max_length-len(A):] == A:
                n_wins[0] += 1
                break
            elif ch_window[max_length-len(B):] == B:
                n_wins[1] += 1
                break

    winrates = n_wins/N
    av_n_flips = n_flips/N
    return winrates, av_n_flips

def all_patterns(j,alphabet):
    '''
    recusively builds all patterns of length j from alphabet
    note: before calling must initialize following two lists within module:
    >>>k=3
    >>>conway.list_pattern=['-']*k
    >>>conway.patterns = []
    >>>conway.all_patterns(k,alphabet)
    >>>patterns = conway.patterns
    '''
    global list_pattern
    global patterns
    if j == 1:
        for key in alphabet.keys():
            list_pattern[-j] = key
            string_pattern = ''.join(list_pattern)
            patterns.append(string_pattern)
    else:
        for key in alphabet.keys():
            list_pattern[-j] = key
            all_patterns(j-1,alphabet)

