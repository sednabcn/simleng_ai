import sys
from simleng_ai.simleng import Simleng

def main():
    script=sys.argv[0]
    try:
        file_input=sys.argv[1]        
    except:
        print("file_input is empty or not adequated file") 
    try:
        score=max(float(sys.argv[2]),0.0)
    except:
        print("score value is None, Re-enter its value")
        score=0.0

    
    Simleng(file_input=file_input, score=score).simulation_strategies()
if __name__ == '__main__':
        main()
        
        """
        python api.py simlengin4.txt 0.90
        """
        
