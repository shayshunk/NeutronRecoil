import pandas as pd

original_df = pd.DataFrame({"foo": range(5), "bar": range(5, 10)})  
original_df.to_pickle("dummy.pkl")  