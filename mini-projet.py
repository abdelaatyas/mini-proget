import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##################################################################################  de linge 5 a ligne 45 creation du'une dataset de 2000 etudiants avec les notes de 6 matieres

class etudient :    # creation de la classe etudient
	def __init__(self, Nom=None, Appogie=None, Age=None, Notes=[]) :
		self.Nom = Nom
		self.Appogie = Appogie
		self.Age = Age
		self.Notes = Notes
		if len(self.Notes) == 6:
			self.Math,self.MachineLearning,self.Python,self.Imp3D,self.ENG,self.SQL=self.Notes
		else:
			self.Math,self.MachineLearning,self.Python,self.Imp3D,self.ENG,self.SQL = [0]*6
etudiant = etudient()
# Initialisation de la liste des etudiants
students = []
# Creation de 2000 etudiants avec des notes aleatoires
for i in range(2000): #
    student = {
        'Nom': "etudient" + str(i+1),
        'Appogie': np.random.randint(210000, 240000),
        'Age': np.random.randint(18, 25),
        'Notes' : np.random.randint(0, 21, size=6).tolist(),
        'Math':0,
        'MachineLearning':0,
        'Python':0,
        'Imp3D':0,
        'ENG':0,
        'SQL':0
    }
    students.append(student)
for i in range(2000):
    students[i]['Math'] = students[i]['Notes'][0]
    students[i]['MachineLearning'] = students[i]['Notes'][1]
    students[i]['Python'] = students[i]['Notes'][2]
    students[i]['Imp3D'] = students[i]['Notes'][3]
    students[i]['ENG'] = students[i]['Notes'][4]
    students[i]['SQL'] = students[i]['Notes'][5]

df = pd.DataFrame(students)

for col in ['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG', 'SQL']:
    df.loc[df.sample(frac=0.1).index, col] = np.nan

df.to_csv("etudient.csv", index=False, columns=['Nom', 'Appogie', 'Age', 'Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG', 'SQL'])
##################################################################################
# Importaion du fichier etudiant.csv
df = pd.read_csv("etudient.csv")
# Affichage des 10 premières lignes du DataFrame
print("Les 10 premieres lignes :\n", df.head(n=10))
# Affichage des 10 dernières lignes du DataFrame
print("Les 10 dernieres lignes :\n", df.tail(n=10))
# Affichage des informations sur le DataFrame
print("les informations sur le dataframe :\nb", df.info())
# Affichage des donnees manquantes:
print("les donnees manquantes :\n", df.isnull().sum())
# Affichage des noms ayants des notes manquantes 
print("les etudiants ayants des notes manquantes :\n", df[df.isnull().any(axis=1)][['Nom', 'Appogie']])
# Affichage des statistiques descriptives du DataFrame
print("Les statistiques descriptives :\n", df.describe())
# Affichage de la moyenne des notes de chaque matiere
print("les moyennes des notes :\n", df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG', 
'SQL']].mean())
# Affichage de la mediane des notes de chaque matiere
print("les medianes des notes :\n", df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG',
'SQL']].median())
# Affichage de l'ecart type des notes de chaque matiere
print("les ecarts types des notes :\n", df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG',
'SQL']].std())
# Affichage de la variance des notes de chaque matiere  
print("les variances des notes :\n", df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG',
'SQL']].var())
# Affichage de la correlation entre les notes de chaque matiere
print("la correlation entre les notes :\n", df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG',
'SQL']].corr())
# Affichage de la matrice de correlation
print("la matrice de correlation :\n", df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG',
'SQL']].corr())
# Affichage de la matrice de correlation sous forme de heatmap
plt.figure(figsize=(10, 8))
plt.title("Matrice de correlation")
plt.imshow(df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG', 'SQL']].corr(), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG', 'SQL']].columns)), df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG', 'SQL']].columns, rotation=45)
plt.yticks(range(len(df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG', 'SQL']].columns)), df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG', 'SQL']].columns)
plt.show()
# Remplacer les valeurs nan par la moyenne de chaque note
for col in ['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG', 'SQL']:
    df[col]=df[col].fillna(df[col].mean())
# Vérification des données après remplacement
print("Les données après remplacement des NaN :\n", df.isnull().sum())
df['Moyenne'] = df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG', 'SQL']].mean(axis=1)
print("la moyenne des notes pour chaque etudiant est : ", df['Moyenne'])
# Trouver l'étudiant avec la moyenne la plus élevée
etudiant_max=df.loc[df['Moyenne'].idxmax()]
print("le premier etudient au niveau d'université est  :\n", etudiant_max)
# Trouver l'étudiant avec la moyenne la plus basse
etudiant_min=df.loc[df['Moyenne'].idxmin()]
print("le dernier etudient au niveau d'université est  :\n", etudiant_min)

# Filter les donnees :

etudiants_succes = df[df['Moyenne'] >= 10]
print("Nombre d'étudiants ayant réussi :", len(etudiants_succes))
etudiants_succes_trie = etudiants_succes.sort_values(by='Moyenne', ascending=False)
# Affichage des premiers résultats
print("Étudiants réussis (les 20 premiers etudients au niveau d'université) :\n", etudiants_succes_trie.head(20))

# visualisation des notes 

plt.figure(figsize=(10,6))
plt.scatter(df['Age'], df['Moyenne'], alpha=0.5, c='blue')
plt.title("Relation entre l'âge et la moyenne des notes")
plt.xlabel("Âge")
plt.ylabel("Moyenne des notes")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(df['Python'], df['MachineLearning'], alpha=0.5 , c='green')
plt.xlabel("les notes python")
plt.ylabel("les notes machine learning ")
plt.title("Relation entre les notes Machine learnig et python ")
plt.show()


plt.hist(df['Moyenne'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution des moyennes des étudiants")
plt.xlabel("Moyenne")
plt.ylabel("Nombre d'étudiants")
plt.grid(True)
plt.show()

plt.scatter(df['Math'], df['Python'], alpha=0.5)
plt.title("Corrélation entre Math et Python")
plt.xlabel("Math")
plt.ylabel("Python")
plt.grid(True)
plt.show()

moyennes = df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG', 'SQL']].mean()

plt.figure(figsize=(8,6))
moyennes.plot(kind='bar', color='green', edgecolor='black')
plt.title("Moyenne des notes par matière")
plt.ylabel("Note moyenne")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df['Age'], bins=range(18, 26), color='orange', edgecolor='black')
plt.title("Distribution des âges des étudiants")
plt.xlabel("Âge")
plt.ylabel("Nombre d'étudiants")
plt.grid(True)
plt.show()

sns.pairplot(df[['Math', 'MachineLearning', 'Python', 'Imp3D', 'ENG', 'SQL']])
plt.suptitle("Relations entre les matières", y=1.02)
plt.show()


age_grouped = df.groupby('Age')['Moyenne'].mean()

plt.figure(figsize=(8,5))
age_grouped.plot(kind='line', marker='o', color='purple')
plt.title("Moyenne des notes selon l'âge")
plt.xlabel("Âge")
plt.ylabel("Moyenne des notes")
plt.grid(True)
plt.show()

bins = [0, 5, 10, 15, 20]
labels = ['[0-5[', '[5-10[', '[10-15[', '[15-20]']
df['Tranche'] = pd.cut(df['Moyenne'], bins=bins, labels=labels, include_lowest=True)

plt.figure(figsize=(7,5))
df['Tranche'].value_counts().sort_index().plot(kind='bar', color='teal')
plt.title("Répartition des étudiants par tranche de moyenne")
plt.xlabel("Tranche de moyenne")
plt.ylabel("Nombre d'étudiants")
plt.grid(axis='y')
plt.show()