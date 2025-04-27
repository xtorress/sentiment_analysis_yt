import nltk
import os

class NLTKSetup:
    def __init__(self, nltk_data_path = "../data/nltk_data"):
        self.nltk_data_path = nltk_data_path
        self.create_nltk_data_folder()

    def create_nltk_data_folder(self):
        "Check that nltk_data exists and add the path of nltk data to nltk.data."
        if not os.path.exists(self.nltk_data_path):
            os.makedirs(self.nltk_data_path)
            
        nltk.data.path.append(self.nltk_data_path)

    def download_resources(self, resources=None):
        "Download the necessary resources if they don't exist."
        if resources is None:
            resources = ["punkt", "stopwords", "averaged_perceptron_tagger"]
        
        for resource in resources:
            try:
                nltk.data.find(resource)
            except:
                nltk.download(resource, download_dir=self.nltk_data_path)