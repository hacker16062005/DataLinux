#Đặc trưng Bộ dữ liệu này bao gồm thông tin của ba loại hoa Iris (một loài hoa lan) khác nhau:
#Iris setosa, Iris virginica và Iris versicolor.
#Mỗi loại bông hoa được đo với dữ liệu là 4 thông tin:
#chiều dài, chiều rộng đài hoa (sepal), và chiều dài, chiều rộng cánh hoa (petal).
tabela = [

[5.1,3.5,1.4,0.2,"Iris-setosa1"],
[4.9,3.0,1.4,0.2,"Iris-setosa2"],
[4.7,3.2,1.3,0.2,"Iris-setosa3"],
[4.6,3.1,1.5,0.2,"Iris-setosa4"],
[5.0,3.6,1.4,0.2,"Iris-setosa5"],
[5.4,3.9,1.7,0.4,"Iris-setosa6"],
[4.6,3.4,1.4,0.3,"Iris-setosa7"],
[5.0,3.4,1.5,0.2,"Iris-setosa8"],
[4.4,2.9,1.4,0.2,"Iris-setosa9"],
[4.9,3.1,1.5,0.1,"Iris-setosa10"],
[5.4,3.7,1.5,0.2,"Iris-setosa11"],
[4.8,3.4,1.6,0.2,"Iris-setosa12"],
[4.8,3.0,1.4,0.1,"Iris-setosa13"],
[4.3,3.0,1.1,0.1,"Iris-setosa14"],
[7.0,3.2,4.7,1.4,"Iris-versicolor1"],
[6.4,3.2,4.5,1.5,"Iris-versicolor2"],
[6.9,3.1,4.9,1.5,"Iris-versicolor3"],
[5.5,2.3,4.0,1.3,"Iris-versicolor4"],
[6.5,2.8,4.6,1.5,"Iris-versicolor5"],
[5.7,2.8,4.5,1.3,"Iris-versicolor6"],
[6.3,3.3,4.7,1.6,"Iris-versicolor7"],
[4.9,2.4,3.3,1.0,"Iris-versicolor8"],
[6.6,2.9,4.6,1.3,"Iris-versicolor9"],
[5.2,2.7,3.9,1.4,"Iris-versicolor10"],
[5.0,2.0,3.5,1.0,"Iris-versicolor11"],
[5.9,3.0,4.2,1.5,"Iris-versicolor12"],
[6.0,2.2,4.0,1.0,"Iris-versicolor13"],
[6.1,2.9,4.7,1.4,"Iris-versicolor14"],
[5.6,2.9,3.6,1.3,"Iris-versicolor15"],
[6.7,3.1,4.4,1.4,"Iris-versicolor16"],
[5.6,3.0,4.5,1.5,"Iris-versicolor17"],
[6.3,3.3,6.0,2.5,"Iris-virginica1"],
[5.8,2.7,5.1,1.9,"Iris-virginica2"],
[7.1,3.0,5.9,2.1,"Iris-virginica3"],
[6.3,2.9,5.6,1.8,"Iris-virginica4"],
[6.5,3.0,5.8,2.2,"Iris-virginica5"],
[7.6,3.0,6.6,2.1,"Iris-virginica6"],
[4.9,2.5,4.5,1.7,"Iris-virginica7"],
[7.3,2.9,6.3,1.8,"Iris-virginica8"],
[6.7,2.5,5.8,1.8,"Iris-virginica9"]]
vetor = [4,2,3,7]
teste=[]
import math
def distanciaEuclidiana(a,b):
    d = 0
    for i,j in zip(a,b):
        d+= (i-j)**2
    return math.sqrt(d)
def vizinhos(vetor,tabela):
    vizs = []
    for i in tabela:
        vizs.append(distanciaEuclidiana(vetor,i[:-1]))
    return vizs


distancias = vizinhos(vetor,tabela)
teste=distancias[:]
teste.sort()
#k là số lần for
for i in range(10):
    posicao = distancias.index(teste[i])
    print(tabela[posicao][-1])
