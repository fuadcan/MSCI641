import sys,re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":
    input_path = sys.argv[1]
    # Reading the file
    comments = open(input_path).readlines()
    comments = [comment.lower() for comment in comments]
    # cleaning for tokenization
    comments = '<<EOS>><<BOS>>'.join(comments)
    comments = comments.replace('&#34;','"')
    filename = re.findall('([A-Za-z0-9_-]+).txt', input_path)[0]

    print('[LOG]: Flagging urls and emails')
    urls = re.findall(r'(http\S*[A-Za-z]+[.]\w{,3}[^\s.)]*)',comments)

    ## Flag abbreviations, emails and websites
    ## For abbreviations
    comments = re.sub(r'(["\s.,!?](([a-zA-Z]\.){2,}[a-zA-Z]?))',r' <<\2>> ',comments) # ([\s.,!?]((?:[a-zA-Z]\.){2,}))

    ## For emails
    comments = re.sub(r'([A-Za-z0-9]*@[A-Za-z0-9]*[.]com)',r' <<\1>> ',comments)

    ## For http links
    comments = re.sub(r'(http\S*[A-Za-z]+[.]\w{,3}[^\s.)]*)',r' <<URL>> ',comments)

    ## For other links
    comments = re.sub(r'([A-Za-z]+[.]com)',r' <<\1>> ',comments)

    ## Replacing urls back
    def callback(match):
        return next(callback.v)
    callback.v=iter(tuple(urls))

    if len(urls) == len(re.findall(r'<<URL>>',comments)):
        comments = re.sub(r'<<URL>>',callback,comments)
        print('[LOG]: Replaced urls back to their placeholders')

    tokens_comments = [re.findall(r"<<.*?>>|[A-Za-z0-9]+|[^\w\s]", comment) for comment in \
              comments.split('<<EOS>><<BOS>>')] # [A-Za-z0-9]+[-_][A-Za-z0-9]+|
    print('[LOG]: Precleaned the file')

    def token_cleaner(token):
        if re.search("[!\"#<>$%&()*+/:;<=>@[\\]^`{|}~\t\n]",token):
            token = re.sub("[!\"#$%&()*+/:;<=>@[\\]^`{|}~\t\n']","",token)
        return(token)

    tokens_comments = [[token_cleaner(token) for token in sentence] for sentence in tokens_comments]
    tokens_comments = [[token for token in sentence if token is not ''] for sentence in tokens_comments]
    print('[LOG]: Cleaned the special characters')


    sws = list(set(stopwords.words('english')))
    tokens_comments_cln = [[t for t in sentence if t.lower() not in sws] for sentence in tokens_comments]
    print('[LOG]: Cleaned stopwords')

    ## Train - Test - Validation
    # with stopwords
    comments_train, comments_test = train_test_split(tokens_comments,  test_size=0.2, random_state=156)
    comments_test, comments_val   = train_test_split(comments_test, test_size=.5, random_state=156)

    # Without stopwords
    comments_train_cln, comments_test_cln = train_test_split(tokens_comments_cln,  test_size=0.2, random_state=156)
    comments_test_cln,  comments_val_cln  = train_test_split(comments_test_cln, test_size=.5, random_state=156)


    print('[LOG]: Saving files')


    np.savetxt(filename + "_train.csv", comments_train, delimiter=",", fmt='%s')
    np.savetxt(filename + "_val.csv",   comments_val,   delimiter=",", fmt='%s')
    np.savetxt(filename + "_test.csv",  comments_test,  delimiter=",", fmt='%s')

    np.savetxt(filename + "_train_no_stopword.csv", comments_train_cln,
               delimiter=",", fmt='%s')
    np.savetxt(filename + "_val_no_stopword.csv", comments_val_cln,
               delimiter=",", fmt='%s')
    np.savetxt(filename + "_test_no_stopword.csv", comments_test_cln,
               delimiter=",", fmt='%s')
    print('[LOG]: Done')
