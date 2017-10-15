import os
import pandas as pd


# Set directory here
directory = 'E:/OneDrive/EM-Search/MJK-SvJ/Raw'


def separate_svj(folder):
    os.chdir(directory)
    os.chdir('./' + folder)
    files = os.listdir()
    for f in files:
        filename = f
        imgnum = filename[6:9]
        df = pd.read_csv(filename, sep = '\t')

        # -- Clean and avg
        avgx = (df.iloc[:, 1] + df.iloc[:, 4]) / 2
        avgy = (df.iloc[:, 2] + df.iloc[:, 5]) / 2
        df['AvgX'] = avgx
        df['AvgY'] = avgy
        # -- Select only rows where validityR/L != 4
        df = df.drop(df[(df.ValidityLeft == 4) | (df.ValidityRight == 4)].index)
        df = df[['ParticipantName', 'AvgX', 'AvgY']] #Select only these columns

        # -- Begin selecting/writing P## to csv
        for p in df.ParticipantName.unique():

            prows = df.loc[(df['ParticipantName'] == p)]  # Select rows with P## = p
            prows.to_csv(p + '-' + imgnum + '.csv', index=False) # write to csv

