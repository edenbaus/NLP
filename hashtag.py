"""
this code is for python 3.x
dependencies are listed in the included requirements.txt file
For desired results, download the spacy english models prior to running this file:
					python3 -m spacy.en.download all

This code does generates a userdefined nmber of hashtags from a series of text files:
	reads in a user defined directory of text files (.txt)
	cleans/sorts the text
	finds and counts occurances of each unique word (and 'similar/plural' word)
	prints the most frequent 'n' words
	collects and prints index of the sentences where they occur
"""
print('*'*75+'\n\nThis Python 3 script will take a direcctory of .txt files\ngenerate hashtags based on word frequency\nofcourse we ignore stopwords\n\nLets Begin!')
print('Please be patient...')
print('Importing libraries')
import spacy
from spacy.en import English
import os
from sys import version_info
import time
from spacy.attrs import ORTH
import random
import numpy as np
import pandas as pd
import textacy
from collections import Counter
import sys

print('Setting language settings to English')
parser=English()
nlp = spacy.load('en')

def get_num_hashtags():
	n = 0
	py3 = version_info[0] > 2 #creates boolean value for test that Python major version > 2
	if py3:
		while n == 0:
			try:
				tmp = int(input("Please Enter the desired amount of hashtags in [1,12] "))
				if tmp < 13 and tmp > 0:
					n = tmp
				else:
					tmp = int(input("Find hashtags: Enter an integer between 1 - 12 "))
					if tmp < 13 and tmp > 0:
						n = tmp
			except:
				print ('n needs to be an int: 0 < n < 13\n using default value: n=5')
				n = 5
	return (n)

#n = get_num_hashtags()	

def get_texts():
    """
    This function reads in all text files in a specified directory.
    First the user is prompted to enter the path of the directory with text files of interest
    Each file is parsed with readline(), all text content is appended to a temporary list.
    List of text are combined/joined to single string per document
    Function returns dictionary (document:text)
    """
    #path=''
    while True: #path == '':
        py3 = version_info[0] > 2 #creates boolean value for test that Python major version > 2
        if py3:
            path_ = input("Please Enter the Directory with the text files: ")
        else:
            print('Wrong version of Python!\nTry again with Python 3.x')
        try:
            if [f for f in os.listdir(path_) if not f.startswith('.')]:    #omitting hidden files ie: filename has leading '.'
                text_files = [f for f in os.listdir(path_) if not f.startswith('.')]
                break
        except:
            print ('bad path!\tTrying default: "./test_docs/"')
            path_ = './test_docs/'    # overwriting user inputted path - temporarily
            if [f for f in os.listdir(path_) if not f.startswith('.')]:    #omitting hidden files ie: filename has leading '.'
                text_files = [f for f in os.listdir(path_) if not f.startswith('.')]
                break

    txt = {}
    count = 1
    num_texts = len(text_files)
    for text in text_files:
        if text[-4:] == '.txt':         # making sure we're working with .txt files
            print ('\n'+'*'*75)
            print ('Reading in file %s \t %d of %d\n' %(str(text),count,num_texts))
            count += 1
            lines = []                  # new list where text is appended
            file_f = path_ + text # full file path for each text file
            f = open(file_f, encoding='utf-8', errors='ignore')
            #f = open(file_f)            # opens text file
            next_ = f.readline()#.decode('latin-1').encode("utf-8")   

            while next_ != "":          # parsing text file
                lines.append(next_)     # adding each f.readline() to the list
                next_ = f.readline()#.decode('latin-1').encode("utf-8")    # parsing text file
            txt[text[:-4]] = ' '.join(lines) # create key/value pair filename:text
    print ('-'*75+'\n\t\t\t\tComplete!\n'+'-'*75+'\n')    
    return (txt)
 
    
def prep_nlp():
    """
    prep spacy stopwords for hashtag creation
    """
    Morestop = [',','let','man','$','=','+','_','/',';','saw','s','?']
    for word in Morestop:
        nlp.vocab[(word)].is_stop = True
    print ('Prepare stopwords...')
    
def prepare_text(document):
    """
    textacy - spacy wrapper
    cleans up text in one go
    -lower case
    -remove numbers
    -remove punctuation
    -undo contractions
    -removes accents
    -fixes garbled unicode
    -removes currency symbols
    -string replace '\n' with '' too
    """
    text_processing = textacy.preprocess_text(
                        nlp(document).text.replace('-',' ').replace('\n',''),
                        fix_unicode=True,
                        lowercase=True,
                        transliterate=False,
                        no_urls=False,
                        no_emails=False,
                        no_phone_numbers=False,
                        no_numbers=True,
                        no_currency_symbols=True,
                        no_punct=True,
                        no_contractions=True,
                        no_accents=True
                    )
    prepared_text = nlp(text_processing)
    print ('cleaning text...')
    return (prepared_text)

def doc_sort(texts):
    """
    sorts the filenames/key from dictionary
    preps for get_hashtag_sents()
    ie: 'doc3', 'doc2', 'doc4', 'doc1'...
    returns ['doc1','doc2','doc3','doc4'...]
    """
    print ('sorting documents...')
    F_sort = sorted(set([key for key in texts.keys()]))
    print ('sorting finished!')
    return (F_sort)

def most_similar(word):
    """
    Approach to dealing with word stems
    find most similar word according to spacy
    return similar word if similarity with origin word > .5 and both 
    words have same starting letter
     
    ex1: in doc1, work has 12 occurances, working has 3, combined 15 occurances
    ex2: in doc1, generation has 11 occurances, generations has 2... 13 occurances
    unfortunately spacy's underlying functionality isn't very rhobust and performs
    quite inconsistantly, particularly with identifying plural forms
    """
    print('Finding similar words for %s...' %(word.text))
    by_similarity = sorted(word.vocab, key=lambda w: word.similarity(w), reverse=True)
    similar_ = [w.orth_ for w in by_similarity if w.is_lower == True][1] #grabs first distinct item in list
    if word.similarity(nlp(similar_)) > .5 and word.text[0] == similar_[0]: #min of 50% similarity and starting letter
        return(similar_)
    else:
        if word.text[~0] == 's':
            return(word.text[:-1])
        else:
            return(word.text + 's')
    
def token_tuple(document):
    """
    generate list of tuples with (word,count) structure
    for vocab in given document
    """
    hashtags = []
    doc = prepare_text(document)
    print ('Counting each word in document...')
    counts = doc.count_by(ORTH)
    for word_id, count in sorted(counts.items(), reverse=True, key=lambda item: item[1]):
        hashtag={}
        if nlp.vocab[nlp.vocab.strings[word_id]].is_stop == False:
                hashtags.append((nlp.vocab.strings[word_id], count))
    return (hashtags)

def topn(document,n):
    """
    generates list of n tuples of (word,count) structure with highest count
    """
    hashtags = []
    doc = prepare_text(document)
    counts = doc.count_by(ORTH)
    print ('Finding top %d words in document...' %(n))
    for word_id, count in sorted(counts.items(), reverse=True, key=lambda item: item[1]):
        hashtag={}
        if nlp.vocab[nlp.vocab.strings[word_id]].is_stop == False:
                hashtags.append((nlp.vocab.strings[word_id], count))
                if len(hashtags) == n:      #initially i used dictionares, but they dont preserve order
                    return (hashtags)
                    
def get_hashtag_df(document,n):
    """
    assemble data frame with relevent info about hashtag words
    includes topn words, counts, similar words (if exist), similar word frequency, and combined count
    """
    doc_vocab = pd.DataFrame(token_tuple(document),columns=['Words','Count'])
    topwords = pd.DataFrame(topn(document,n),columns=['Words','Count'])['Words'].tolist()
    #topwords_count = pd.DataFrame(topn(texts['doc1'],5),columns=['Words','Count'])['Count'].tolist()

    print ('building dataframe of hashtags...')
    checkwords = pd.DataFrame(token_tuple(document),columns=['Word','Count'])['Word'][n:].tolist()
    combo_topwords = pd.DataFrame([(topwords[i],most_similar(nlp.vocab[topwords[i]])) for i in range(len(topwords))],columns=['baseword','compword'])
    combo_topwords['Count'] = pd.DataFrame(topn(document,n),columns=['Words','Count'])['Count'].tolist()
    combo_topwords['multiword'] =[combo_topwords['compword'].tolist()[i] in checkwords for i in range(len(combo_topwords))]
    combo_topwords['compword_count'] = pd.Series([doc_vocab[doc_vocab['Words'] == combo_topwords['compword'].tolist()[i]]['Count'].values for i in range(len(combo_topwords))]).map(lambda x:x[0] if len(x) > 0 else 0)
    combo_topwords['total_count'] = combo_topwords['Count'].astype(int) + combo_topwords['compword_count'].astype(int)
    combo_topwords = combo_topwords.sort_values(by='total_count',ascending=False)
    return (combo_topwords)
    

def locate_hashtag(hashtagdf,ind,texts):
	"""
	This function is passed the results from the get_hashtag_df() function
	Using that dataframe, it iterates through the provided hashtags - aka baseword, also next similar word is checked - compword
	First we loop through each sentence in the given text
	Next we loop hrough the provided hashtags - aka baseword, also next similar word is checked - compword
	If the given word is in the given sentence, the sentence is printed, and the index (sentence #)  with the word is stored to a dict object
	the dictionary is then appended to a list and then changed to a DataFrame.
	the resulting dataframe is the return object - it isn't used but available for continued development or use as library
	"""
    print ('-'*75)
    print ('\nlocating sentences where hashtags occur for document %d ...\nThis may take a moment...' %(ind))
    ind = int(ind)
    w = nlp(texts[doc_sort(texts)[ind]])
    doc_sents = [sent for sent in w.sents]
    doc_sents_c = [str(doc_sents[i]).lower().replace('-','').replace('\n','').replace('.','').replace(',','').replace('?','').replace('.','') for i in range(len(doc_sents))]

    #get dataframe of words & counts
    dft = hashtagdf[ind]#sorted_hashtag_df[ind]
    doc_hashtags_df = dft[['baseword','total_count']]
    doc_hashtags_df.columns=['hashtag','count']
    locationlist = []

    print('Document: %s\t hashtags:%s\n' %(doc_sort(texts)[ind],sorted(dft['baseword'].tolist())))
    print('-'*75 + '\n')
    print(doc_hashtags_df)
    print('\n')
    
    for i in range(len(doc_sents_c)):
        posdict = {}
        for j in range(len(dft['baseword'])):
            if dft['baseword'][j] in doc_sents_c[i]:
                posdict[dft['baseword'][j]] = i
                print ('%s -\nsentence: %d\n \t %s\n' %(dft['baseword'][j],i,doc_sents[i]))
            elif dft['multiword'][j] == True and dft['compword'][j] in doc_sents_c[i]:
                posdict[dft['baseword'][j]] = i
                print ('%s - sentence: %d \t %s\n' %(dft['baseword'][j],i,doc_sents[i]))
        locationlist.append(posdict)
    print('Document: %s\t' %(doc_sort(texts)[ind]))
    print('-'*75)
    return(pd.DataFrame(locationlist).replace(np.nan,'',regex=True))

def main():
	"""
	pretty straight forward main() function
	acts as a location to setup 'scaffolding' and trigger the locate_haghtag() function
	"""

	n = get_num_hashtags()	
	texts = get_texts()	
	prep_nlp()
	#hashtag_df = [get_hashtag_df(texts[i]) for i in texts.keys()]
	try:
		sorted_hashtag_df = [get_hashtag_df(texts[doc_sort(texts)[i]],n) for i in range(len(texts.keys()))]	
		[locate_hashtag(sorted_hashtag_df,i,texts) for i in range(len(texts))]
	except:
		print('looks like the text file names dont end in a numeric value... lets try that again')
		#hashdf = [get_hashtag_df(texts,n) for i in range(len(texts.keys()))]
		#[locate_hashtag(hashdf,i,
		
if __name__ == "__main__":
	main()
