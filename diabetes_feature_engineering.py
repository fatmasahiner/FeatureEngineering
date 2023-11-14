
# Gerekli kütüphane ve fonksiyonlar

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.simplefilter(action="ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.max_rows", 20)
pd.set_option("display.float_format", lambda x: "%3f" % x)

df = pd.read_csv("6_hafta_Feature_Engineering/datasets/diabetes.csv")


######################################################################################
# Görev 1 : Keşifçi Veri Analizi
#######################################################################################

# Adım 1: Genel resmi inceleyiniz.

def check_df(dataframe, head=5):
    print("############### Shape ###############")
    print(dataframe.shape)
    print("############### Types ###############")
    print(dataframe.dtypes)
    print("############### Head ###############")
    print(dataframe.head(head))
    print("############### Tail ###############")
    print(dataframe.head(head))
    print("############### NA ###############")
    print(dataframe.isnull().sum())
    print("############### Quantiles ###############")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
def grab_col_names(dataframe, cat_th=10, car_th=20 ):
    """
    Veri setindeki kategorik ,numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    Parameters:
        dataframe: dataframe
                    Değiken isimleri alınmak istenilen dataframe

        cat_th: int,optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri

        car_th: int,optional
                kayegorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns:

        cot_cols:list
                Kategorik değişken listesi
        num_cols:list
                Numerik değişken listesi
        cat_but_car:list
                Kategorik görünümlü kardinal değişken listesi


        Examples

            import seaborn as sns
            df=sns.load_dataset("iris")
            print(grab_col_names(df))

         Notes

            cat_cols+num_cols+cat_but_car= Toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde

    """

#cat_cols, cat_but_car

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "0"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "0"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() < car_th and dataframe[col].dtypes == "0"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in  cat_cols if col not in cat_but_car]

#num_cols

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "0"]
    num_cols = [col for col in num_cols if col not in num_but_cat]


    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols:{len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car:{len(cat_but_car)}")
    print(f"num_but_cat:{len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car

#kategorik değişkenlerin analizi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100*dataframe[col_name].value_counts()/len(dataframe)}))
    print("##############################")
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()

cat_summary(df, "Outcome")

#numerik değişkenlerin analizi

def num_summery(dataframe,numerical_col, plot=False):
    quantiles=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summery(df,col, plot=True)


# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

#Numerik değişkenlerin target analizi

def target_summary_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}),end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df,"Outcome", col)




# Adım 6: Eksik gözlem analizi yapınız.

df.isnull().sum()

# Adım 7: Korelasyon analizi yapınız.
df.corr()

#korelasyon matrisi

f, ax =plt.subplots(figsize=[18,13])
sns.heatmap(df.corr(), annot = True, fmt=".2f",ax=ax, cmap="magma")
ax.set_title("Corelation Metrix",fontsize=20)
plt.show()

#base model kurulumu
y=df["Outcome"]
X=df.drop("Outcome",axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)


rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)

y_pred = rf_model.predict(X_test)


print(f"Accuaracy:{round(accuracy_score(y_pred, y_test),2)}")
print(f"Recall:{round(recall_score(y_pred, y_test),3)}")
print(f"Precision:{round(precision_score(y_pred, y_test),2)}")
print(f"F1:{round(f1_score(y_pred, y_test),2)}")
print(f"Auc:{round(roc_auc_score(y_pred, y_test),2)}")


def plot_importance(model,features, num=len(X),save=False):
    feature_imp=pd.DataFrame({"Value":model.feature_importances_,"Feature":features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value",y="Feature",data=feature_imp.sort_values(by="Value", ascending=False)[0:num])

    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
       plt.savefig("importances,png")

plot_importance(rf_model,X)
#########################################################################################
# Görev 2 : Feature Engineering
#########################################################################################

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

#not:bir insanda pregnancies ve outcome dışındaki değişken değerleri 0 olamacağı bilinir Bu yüzden bu değerlerle ilgili aksiyon kararı olarak 0 olan değerlere NaN atanabilir.

zero_columns=[col for col in df.columns if (df[col].min()==0 and col not in ["Pregnancies","Outcome"])]
zero_columns

#Gözlem birimlerinde 0 olan değişkenlerin her birisine gidip 0 olan gözlem birimlerini NaN ile değiştirdik

for col in zero_columns:
    df[col]=np.where(df[col] == 0, np.nan, df[col])

     #Eksik gözlem analizi
df.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

#Eksik değerlerin bağımlı değişken ile ilişkisinin incelenmesi

def missing_vs_target(dataframe, target, na_columns):
    temp_df=dataframe.copy()
    for col in na_columns:
        temp_of=dataframe.copy()
        for col in na_columns:
            temp_df[col +"_NA_FLAG"]=np.where(temp_df[col].isnull(),1,0)
    na_flags= temp_df.loc[:,temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN":temp_df.groupby(col)[target].mean(),
                            "Count":temp_df.groupby(col)[target].count()}),end="\n\n\n")
missing_vs_target(df,"Outcome",na_columns)

#Eksik değerlerin oluşturulmasında

for col in zero_columns:
    df.loc[df[col].isnull(),col] = df[col].median()

df.isnull().sum()
df.head(20)

#Aykırı Değer Analizi

def outlier_threshoulds(dataframe,col_name, q1=0.05, q3=0.95):
    quartile1= dataframe[col_name].quantile(q1)
    quartile3= dataframe[col_name].quantile(q3)
    interquantile_range=quartile3-quartile1
    up_limit= quartile3+1.5*interquantile_range
    low_limit=quartile1-1.5*interquantile_range
    return low_limit,up_limit

def check_outlier(dataframe,col_name):
    low_limit, up_limit = outlier_threshoulds(dataframe,col_name)
    if dataframe[(dataframe[col_name]>up_limit)|(dataframe[col_name]<low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable,q1=0.05,q3=0.95):
    low_limit,up_limit=outlier_threshoulds(dataframe, variable, q1=0.05,q3=0.95)
    dataframe.loc[(dataframe[variable]<low_limit),variable]=low_limit
    dataframe.loc[(dataframe[variable]>up_limit),variable]=up_limit


#Aykırı değer analizi ve baskılama işlemi

for col in df.columns:
    print(col, check_outlier(df,col))
    if check_outlier(df, col):
        replace_with_thresholds(df,col)

for col in df.columns:
     print(col,check_outlier(df,col))


# Adım 2: Yeni değişkenler oluşturunuz.

##Özellik çıkarımı

#Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
df.loc[(df["Age"]>=21)&(df["Age"]<50),"NEW_AGE_CAT"]="mature"
df.loc[(df["Age"]>=50),"NEW_AGE_CAT"]="senior"


#BMI 18.5 aşagısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası owerwight ve 30 üsü obez
df["NEW_BMI"] = pd.cut(x=df["BMI"],bins=[0,18.5,24.9,29.9,100],
                     labels=["Underweight","Healthy","Overweight","Obese"])


#Glukoz değerini kategorik değişkene çevirme
df["NEW_GLUCOSE"]=pd.cut(x=df["Glucose"],bins=[0,140,200,300], labels=["Normal","Prediabetes","Diabetes"])

####Yaş ve beden kitle indeksini bir arada düşünerekkategorik değişken oluşturma

df.loc[(df["BMI"]<18.5)&((df["Age"]>=21)&(df["Age"]<50)),"NEW_AGE_BMI_NOM"]="underweightmature"
df.loc[(df["BMI"]<18.5)&(df["Age"]>=50),"NEW_AGE_BMI_NOM"]="underweightsenior"

df.loc[((df["BMI"]>=18.5)&(df["BMI"]<25))&((df["Age"]>=21)&(df["Age"]<50)),"NEW_AGE_BMI_NOM"]="healthymature"
df.loc[((df["BMI"]>=18.5)&(df["BMI"]<25))&((df["Age"]>=21)&(df["Age"]>=50)),"NEW_AGE_BMI_NOM"]="healthysenior"

df.loc[((df["BMI"]>=25)&(df["BMI"]<30))&((df["Age"]>=21)&(df["Age"]<50)),"NEW_AGE_BMI_NOM"]="overweightmature"
df.loc[((df["BMI"]>=25)&(df["BMI"]<30))&(df["Age"]>=50),"NEW_AGE_BMI_NOM"]="owerweightsenior"

df.loc[((df["BMI"]>18.5)&((df["Age"]>21))&(df["Age"]<50)),"NEW_AGE_BMI_NOM"]="obesemature"

#Yaş ile glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma

df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "Lowmature"

df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "Lowsenior"

df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] < 100) & (df["Age"] >=21)&(df["Age"]<50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"

df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"

df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"]="hiddenmature"

df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df[ "Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"]="hiddensenior"

df.loc[(df["Glucose"]> 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"

df.loc[(df["Glucose"]> 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"

#İnsulin Değeri ile Kategorik değişken türetnek

def set_insulin (dataframe, col_name="Insulin"):
    if 16 <=dataframe[col_name]<=166:
        return "Normal"

    else:
        return "Abnormal"

#İnsulin değeri ile kategorik değişken üretmek

    df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

    df["NEW_GLUCOSE*INSULIN"]=df["Glucose"]*df["Insulin"]
    df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

    # Kolonlaran büyültülmesi

    df.columns = [col.upper() for col in df.columns]

    df.head()
    df.shape
# Adım 3: Encoding işlemlerini gerçekleştiriniz.

cat_cols, num_cols, cat_but_car=grab_col_names(df)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
#Label encoding

def label_encoder(dataframe, binary_col):
    labelencoder =LabelEncoder()
    dataframe[binary_col]=labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)


df.head()

# One-Hot Encoding İşlemi
#cat_cols listesinin güncelleme işlemi

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df=one_hot_encoder(df,cat_cols,drop_first=True) #true diyerek dummi tuzağına düşmedik

df.shape
df.head()

#Standartlaştırma

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape


# Adım 5: Model oluşturunuz.


#Modelleme

y=df["OUTCOME"]
X=df.drop("OUTCOME", axis=1)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.30,random_state=17)

######################################################
rf_model=RandomForestClassifier(random_state=46).fit(X_train,y_train)
y_pred=rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test),2)}")
print(f"Recall:{round(recall_score(y_pred, y_test),3)}")
print(f"Precision:{round(precision_score(y_pred, y_test),2)}")
print(f"F1:{round(f1_score(y_pred, y_test),2)}")
print(f"Auc:{round(roc_auc_score(y_pred, y_test),2)}")

#Accuracy:0.77
#Recall:0.706
#precision:0.59
#F1:0.64
#Auc:0.75


#yeni model başarısı
#Accuracy: 0.79
#Recall:0.732
#Precision:0.64
#F1:0.68
#Auc:0.78

################################# Feature Importance ######################################

def plot_impotance(model,features,num=len(X),save=False):
    feature_imp=pd.DataFrame({"Value":model.feature_importances_,"Feature":features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value",y="Feature",data=feature_imp.sort_values(by="Value",ascending=False)[0:num])


    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_impotance(rf_model, X)