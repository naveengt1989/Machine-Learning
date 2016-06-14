
print ("checking for nltk")
try:
    import nltk
except ImportError:
    print ("you should install nltk before continuing")

print ("download will complete at about 423 MB")
import urllib
url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz"
urllib.urlretrieve(url, filename="../enron_mail_20150507.tgz") 
print ("download complete!")


#import tarfile
#import os
#os.chdir("..")
#tfile = tarfile.open("enron_mail_20150507.tgz", "r:gz")
#tfile.extractall(".")

