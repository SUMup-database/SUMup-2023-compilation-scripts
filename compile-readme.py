import pandas as pd
import codecs
# %% updating author list
sheet_id = "1EdShh5Wg8luE2ifn2oz0GfneixOwGGe8acV_VZWsfuo"
sheet_name = "confirmed"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

df = pd.read_csv(url)
df = df.loc[:df['first name'].last_valid_index(),:]
df = df.iloc[:-1,:]

df_authors = df[['first name', 'last name', 'Affiliation N.']]
df_authors['Affiliation N.']=df_authors['Affiliation N.'].copy().astype(int).astype(str)
df_authors.loc[df_authors.index[df_authors.iloc[:,0].isnull()]-1, 'Affiliation N.'] = \
    (df_authors.loc[df_authors.index[df_authors.iloc[:,0].isnull()]-1, 'Affiliation N.'].values +', '+df_authors.loc[df_authors.index[df_authors.iloc[:,0].isnull()], 'Affiliation N.'].values)

df_authors.loc[df_authors.index[df_authors.iloc[:,0].isnull()]-1, 'Affiliation N.']
df_authors.drop(index=df_authors.index[df_authors.iloc[:,0].isnull()])
file_to_delete = open("doc/ReadMe_2023_src/author_list.tex",'w')
file_to_delete.close()

f = codecs.open("doc/ReadMe_2023_src/author_list.tex", "a", "utf-8")
for ind, row in df_authors.iterrows():
    # row  = [urllib.parse.quote(s) for s in row]
    # if ind == 1: break
    if not isinstance(row[0],str):
        continue
    f.write("\\Author[%s]{%s}{%s}\n"%(row[2],row[0],row[1]))

f.write("\n")
df_affil = df.iloc[df.iloc[:,14].first_valid_index():df.iloc[:,14].last_valid_index(),14].reset_index(drop=True)
for ind, row in enumerate(df_affil.values):
    f.write("\\affil[%i]{%s}\n"%(ind+1,row))


f.close()

# %% compiling latex file
import os
import shutil  
os.chdir('doc/ReadMe_2023_src/')
os.system("pdflatex SUMup_2023_ReadMe.tex")
shutil.move('SUMup_2023_ReadMe.pdf', '../../SUMup 2023 beta/SUMup_2023_ReadMe.pdf')

# cleanup
os.remove('SUMup_2023_ReadMe.toc')
os.remove('SUMup_2023_ReadMe.aux')
os.remove('SUMup_2023_ReadMe.out')
