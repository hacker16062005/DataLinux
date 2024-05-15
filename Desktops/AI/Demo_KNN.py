import math 
def phanloai(points,p,k=3):
    distance=[]
    for group in points:
        for feature in points[group]:
 
            #Tinh khoang cach Euclide 
            euclidean_distance = math.sqrt((feature[0]-p[0])**2 +(feature[1]-p[1])**2)
 
            # Them khoang cach vao mang
            distance.append((euclidean_distance,group))
 
    # Sap xep lai mang va chon k phan tu trong mang co khoang cach min
    distance = sorted(distance)[:k]
 
    freq1 = 0 #Phan loai thuoc nhom 0
    freq2 = 0 #Phan loai thuoc nhom 1
 
    for d in distance:
        if d[1] == 0:
            freq1 += 1
        elif d[1] == 1:
            freq2 += 1
 
    return 0 if freq1>freq2 else 1
 
# Chuong trinh chinh
def main():
# Gán bo du lieu thuoc 2 nhom 0 va 1
 
    points = {0:[(3,11),(2,6),(11,4),(3,24),(14,2),(23,4),(14,16)],
              1:[(2,17),(4,23),(1,12),(6,2),(9,4),(24,23),(7,6),(12,3)]}
 
    # Du lieu can test
    p = (3,9)
 
    # So luong lang giang lan can k
    k = 5
 
    print("Gia tri test thuoc nhom: {}".\
          format(phanloai(points,p,k)))
 
if __name__ == '__main__':
    main()
