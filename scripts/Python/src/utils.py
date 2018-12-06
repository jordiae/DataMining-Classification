from sklearn import preprocessing


def convert_cat_num(dades):
    les = {}
    obj_df = dades.select_dtypes(include=['object']).copy()
    for i in list(obj_df):
        lb = preprocessing.LabelEncoder()
        dades[i] = lb.fit_transform(obj_df[i])
        les[i] = lb
    print(dades.head())
    print(les)
    return les  # retorna el mapa per tornar a convertir de num a cat
