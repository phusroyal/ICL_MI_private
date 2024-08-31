from icl_mi import mutual_Token

def main():
    #Initialize the model
    print('Initializing the model')
    mt = mutual_Token.mutualToken()

    #Print the model
    # print(mt.get_model())

    # calculate the information
    print('Calculating the information')
    mt.calc_information()

if __name__ == '__main__':
    main()