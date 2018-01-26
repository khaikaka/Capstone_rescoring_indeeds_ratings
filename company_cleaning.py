import pandas as pd

def remove_duplicates(df):
    df1 = df.drop(columns=['Unnamed: 0'], axis=1)
    return df1.drop_duplicates(subset = ['user_ids','review_titles'],keep ='first')

def insert_null_text(df, uid):
    for idx, row in df.iterrows():
        if row['user_ids'] == uid:
            upper = df[:idx]
            lower = df[idx:]
            lower.text_reviews = lower.text_reviews.shift(-1)
            lower = lower[:-1]
    df = pd.concat([upper,lower], axis=0)
    return df

def google_cleaning(data_path, uid):
    df = pd.read_csv(data_path)
    df = remove_duplicates(df)
    df = insert_null_text(df, uid)
    return df

def export_data_to_csv(data_path, company, topic, uid):
    df = google_cleaning(data_path, uid)
    file_name = '/Users/hatran/project/galvanize/capstone/temp_data/' + company + '_' + topic + '_data.csv'
    new_name = company + '_' + topic + '_data.csv'
    df.to_csv(file_name)



if __name__ == '__main__':
    uid = 'cmp-review-bd6178c41f6dc094'
    data_path = 'data/Google_data.csv'
    company = 'Google'
    topic = 'all'
    export_data_to_csv(data_path, company, topic, uid)
