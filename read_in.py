import pandas as pd

def read_goodreads(file_ending):
    img_path = "img_"+file_ending
    file_path = "g_"+file_ending+".jl"
    df = pd.read_json(file_path, lines = True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.keys())
        print(df.loc[:,['author']])

def main():
    read_goodreads("0_1_100")

if __name__ == "__main__":
    main()
