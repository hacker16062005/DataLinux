# Ví dụ Phan loại điểm thuộc nhóm 0 hoặc 1
#Sử dụng K nearest neighbour algorithm.
import math
def classifyAPoint(points,p,k=3):
	'''
	Hàm phân loại nhóm cho p thuộc nhóm 0,1.
	Thông số truyền vào-
		points: Bộ điểm đào tạo có 2 loại nhóm 0,1
		p : điểm cần kiểm tra có dạng (x,y)
		k : số hàng xom lân cân gần nhất giả sử k= 3
	'''
	distance=[]  # list lưu khoảng cách từ p tới các điểm trên bộ dữ liệu
	for group in points:
		for feature in points[group]:
			#Tính khoảng cách Euclide từ p tới các điểm huấn luyện
			euclidean_distance = math.sqrt((feature[0]-p[0])**2 +(feature[1]-p[1])**2)
			# thêm mẫu (khoảng cách, nhóm) vào danh sách
			distance.append((euclidean_distance,group))
	# Săp xếp lại list khoảng cách và chọn lựa k khoảng cách min
	distance = sorted(distance)[:k]
	freq1 = 0 #tần số của nhóm 0
	freq2 = 0 #tấn số của nhóm 1
	for d in distance:
		if d[1] == 0:
			freq1 += 1
		elif d[1] == 1:
			freq2 += 1
	return 0 if freq1>freq2 else 1
def main():
	# tìm điểm huấn luyện - 0 and 1
	points = {0:[(1,3),(2,5),(3,6),(3,5),(3.5,4),(2,8),(2,9),(1,7)],
			1:[(5,9),(3,8),(1.5,9),(7,9),(6,7),(3.8,8),(5.6,9),(4,5),(2,5)]}
	# điểm p(x,y) cần kiểm tra
	p = (2.5,7)
	# Số lượng điểm lân cận gần
	k = 3
	print("Kết quả giá trị của p thuộc lớp: {}".\
		format(classifyAPoint(points,p,k)))
if __name__ == '__main__':
	main()
