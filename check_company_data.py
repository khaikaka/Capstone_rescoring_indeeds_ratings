from data_cleaning_before_modeling import remove_duplicates

import pandas as pd
class print_raw_data_row():
    """docstring for ."""
    def __init__(self, company):
        self.company = company
        file_path = 'data/' + self.company + '_all_data.csv'
        data = pd.read_csv(file_path)
        data = remove_duplicates(data)
        data = pd.read_csv(file_path)
        self.data = data
        self.new_data = self.data

    def check_null(self):
        print(list(self.data['user_ids'][pd.isnull(self.data['text_reviews'])]))

    def print_out_row_title(self, title):
        # file_path = 'data/' + self.company + '_all_data.csv'
        # data = pd.read_csv(file_path)
        # data = remove_duplicates(data)
        # data = self.data
        for idx, row in self.data.iterrows():
            if row['review_titles'] == title:
                print(row)

    def print_out_row_uid(self, uid):
        # file_path = 'data/' + self.company + '_all_data.csv'
        # data = pd.read_csv(file_path)
        # data = remove_duplicates(data)
        for idx, row in self.data.iterrows():
            if row['user_ids'] == uid:
                print(idx, row)

    def print_out_row_idx(self, idx):
            # file_path = 'data/' + self.company + '_all_data.csv'
            # data = pd.read_csv(file_path)
            # data = remove_duplicates(data)
            print(self.data.iloc[idx])

    # def insert_null_text(self, uid):
    #     for idx, row in self.data.iterrows():
    #         if row['user_ids'] == uid:
    #             upper = self.data[:idx]
    #             lower = self.data[idx:]
    #             lower.text_reviews = lower.text_reviews.shift(-1)
    #             lower = lower[:-1]
    #     self.new_data = pd.concat([upper,lower], axis=0)
    #

#microsoft.print_out_row('Great fun place to work and learn about new technology')
