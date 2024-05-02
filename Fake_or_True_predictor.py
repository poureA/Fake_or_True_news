#Importing needed modules.

from pickle import load
from tkinter import Tk , Entry , Button , Label
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class Predictor(object):
    def __doc__(self)->str:
        return '''
                 A Tkinter object that get a text as input 
                 then preprocess and predicts the category by three methods.
                 --------------------------------------
                 Filter: Removes the stop words from input.
                 
                 Predict method: preprocess and predicts the category.
                 
                 GUI method: Tkinter GUI for shwowing the result.
                 '''
    
    def Filter(self,text)->str:
        '''
        Remove stop words from input.
        
        -------------------------------
        
        return -> str
        '''
        
        stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
        result = [word for word in text.lower().split() if word not in stopwords]
        return ' '.join(result)
    
    def Predict(self,note):
        '''
        preprocess and predict the category.
        
        -------------------------------------
        
        return -> str
        '''
        
        #Opening the saved Tokenizer object.
        
        with open("D:\\datasets\\Fake_and_True_news\\Tokenizer.pkl",'rb') as file:
            tknzr = load(file)
            
            #Loading the saved Tensorflow model.
            
            model = load_model("D:\\datasets\\Fake_and_True_news\\model")
            
            #Removing stop words from input text.
            
            filtered = self.Filter(note)
            
            #Encoding and making a sequence from input text words.
            
            sequence = tknzr.texts_to_sequences([filtered])
            
            #Padding the sequence.
            
            matrix = pad_sequences(sequence,padding='post',truncating='post',maxlen=16)
            
            #Predicting the category.
            
            pred = model.predict(matrix)
            
            #Generating the result.
            
            if pred[0]>=0.5: return 'True'
            
            else : return 'Fake'
            
    def GUI(self):
        '''
        
        Tkinter GUI for showing the result of Predict method.
        
        '''
        root = Tk()
        
        entry = Entry(font=(20))
        entry.grid(row=1,column=1,ipadx=200,ipady=20)
        
        def Do():
            val = self.Predict(entry.get())
            if val == 'True':
                label = Label(text='True',font=(20),bg='green',fg='white')
                label.grid(row=3,column=1,ipadx=45,ipady=20)
                return label
            else :
                label = Label(text='Fake',font=(20),bg='red',fg='white')
                label.grid(row=3,column=1,ipadx=45,ipady=20)
                return label
            
        btn = Button(text='Predict',bg='yellow',command=Do)
        btn.grid(row=2,column=1,ipadx=50,ipady=10)
        
        label = Label(text='Result',font=(20))
        label.grid(row=3,column=1,ipadx=45,ipady=20)
        
        root.mainloop()

#Instancing and running the class.

predictor = Predictor()
predictor.GUI()
