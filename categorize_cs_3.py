import pandas as pd

'''
surprise: surprise
positive: happiness
negative: CASME II(disgust fear sadness repression) SAMM(disgust fear sadness repression anger)
'''

# CASME II
root_video_dir = '../CASME2/Cropped-updated/Cropped/'
surprise_class = ['surprise']
positive_class = ['happiness']
negative_class = ['disgust', 'fear', 'sadness', 'repression']

categorize_sh = open('categorize_casme_3.sh', 'w')
label_file = open('data_label.txt', 'w')
df = pd.read_excel('../CASME2/CASME2-coding-20190701.xlsx', sheet_name='Sheet1')
video_data = df.iloc[:, [0, 1, 3, 4, 5, 7, 8]].values


for i in range(len(video_data)):
    video_subject = str(video_data[i][0]).zfill(2)
    video_dir = video_data[i][1]
    video_class = video_data[i][-1]
    video_onset = str(video_data[i][2])
    video_apex = str(video_data[i][3])
    video_offset = str(video_data[i][4])
    video_AU = str(video_data[i][-2])

    video_casme2_class = ''
    if video_class in surprise_class:
        video_casme2_class = 'surprise'
    elif video_class in positive_class:
        video_casme2_class = 'positive'
    elif video_class in negative_class:
        video_casme2_class = 'negative'
    else:
        # video_casme2_class = 'others'
        continue
    categorize_sh.writelines('cp -r {} {}\n'.format(root_video_dir + 'sub' + video_subject + '/' + video_dir, './data_3/' + video_casme2_class))
    label_file.writelines('{}\t{}\t{}\t{}\t{}\n'.
                          format('data_3/' + video_casme2_class + '/' + video_dir,
                                 video_onset, video_apex, video_offset, video_AU))


# SAMM
root_video_dir = '../SAMM_cropped/'
surprise_class = ['Surprise']
positive_class = ['Happiness']
negative_class = ['Disgust', 'Fear', 'Sadness', 'Repression', 'Anger']

categorize_sh = open('categorize_samm_3.sh', 'w')
label_file = open('data_label.txt', 'a')
df = pd.read_excel('../SAMM.xlsx', sheet_name='MICRO_ONLY')
video_data = df.iloc[:, [1, 3, 4, 5, 8, 9]].values


for i in range(len(video_data)):
    video_dir = video_data[i][0]
    video_class = video_data[i][-1]
    video_onset = str(video_data[i][1])
    video_apex = str(video_data[i][2])
    video_offset = str(video_data[i][3])
    video_AU = str(video_data[i][-2])

    video_samm_class = ''
    if video_class in surprise_class:
        video_samm_class = 'surprise'
    elif video_class in positive_class:
        video_samm_class = 'positive'
    elif video_class in negative_class:
        video_samm_class = 'negative'
    else:
        # video_samm_class = 'others'
        continue
    categorize_sh.writelines('cp -r {} {}\n'.format(root_video_dir + video_dir, './data_3/' + video_samm_class))
    label_file.writelines('{}\t{}\t{}\t{}\t{}\n'.
                          format('data_3/' + video_samm_class + '/' + video_dir,
                                 video_onset, video_apex, video_offset, video_AU))

