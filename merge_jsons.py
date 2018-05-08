#Run this to merge the files from different download sessions
#The files should be in a 'data' folder and will be returned as 'final'
import glob
from os import listdir

GD_PATH = '.\data'
OUTPUT_NAME = input("Output file name (without .jl extension)? : ") + '.jl'


def read_jsons(jsons, outfile):
    for js_file in jsons:
        with open(js_file) as f:
            lines = f.readlines()
            with open(outfile, "a") as f1:
                f1.writelines(lines)


dirs = os.listdir(GD_PATH)
jsons = glob.glob(GD_PATH + '\*.jl')
read_jsons(jsons, OUTPUT_NAME)
