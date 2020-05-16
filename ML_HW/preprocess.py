def prep_data(df):

    df = df.assign(hw=df["Height"] * df["Width"])
    df = df.assign(vol = (df["Height"]* df["Width"])*df["Length1"])
    
    X = df[["Height", "Width", "hw", "vol"]].values
    y = df["Weight"].values

    return X, y