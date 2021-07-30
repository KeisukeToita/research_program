import csv

class Logger:
    
    def __init__(self):
        self.reward_log=[]
        self.experience_log=[]
        self.file_name=None
        self.header_label=None

    def add_experience(self, data_dict):
        self.experience_log.append(data_dict)
        return

    def set_file_name(self, file_name):
        self.file_name = file_name
        return

    def set_header_label(self, header_label):
        self.header_label = header_label

    def write_csv(self):
        if self.file_name == None:
            raise Exception("file name is None!!")
        elif self.header_label == None:
            raise Exception("header label is None!!")

        with open(self.file_name, 'w', encoding='shift-jis', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows()