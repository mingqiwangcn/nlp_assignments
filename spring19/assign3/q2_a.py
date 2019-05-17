import hmm
def main():
    hmm.prepare()
    hmm.print_dev_log_prob(hmm.dev_data)
    
if __name__ == '__main__':
    main()