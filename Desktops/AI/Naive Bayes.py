print('\n *-----* Classification using Naïve bayes *-----* \n')
total_documents = int(input("Enter the Total Number of documents: "))
                      

                       doc_class = []
                      

                       i = 0
                      

                       keywords = []
                      

                       while not i == total_documents:
                      

                       	doc_class.append([])
                      

                       	text = input(f"\nEnter the text of Doc-{i+1} : ").lower()
                      

                       	class = input(f"Enter the class of Doc-{i+1} : ")
                      

                       	doc_class[i].append(text.split())
                      

                       	doc_class[i].append(class)
                      

                       	keywords.extend(text.split())
                      

                       	i = i+1
                      

                       keywords = set(keywords)
                      

                       keywords = list(keywords)
                      

                       keywords.sort()
                      

                       to_find = input("\nEnter the Text to classify using Naive Bayes: ").lower().split()
                      
