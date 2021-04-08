
single = {
    'x':[],
    'y':[],
    'z':[],
    }

def test():
    import numpy as np
    import csv
    # data = np.loadtxt("3dpoint/3dpoint1.txt")
    # csvFile = open('3dpoint/3dpoint1.txt', 'r')
    # reader = csv.reader(csvFile)
    # 3dpoint/3dpoint1.txt
    with open("test.txt", 'r') as f:
        data = f.readlines()
        for line in data:
            numbers = line.split(',')
            print(numbers)
            numbers_float = list(map(float, numbers))
            print(numbers_float)
            single['x'].append(numbers_float[0])
            single['y'].append(numbers_float[1])
            single['z'].append(numbers_float[2])
        print(single)
    # print(reader)

def draw():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax2 = Axes3D(fig)
    
    import numpy as np
    
    ax2.scatter3D(single['x'],single['y'],single['z'])
    plt.show()


class Position():
    def __init__(self, type, length):
        self.type = type # Hips Spine ...
        self.length = length # length of frames


class Action():
    def __init__(self,filepath):
        self.df = None
        self.filepath = filepath

        self.parse_csv()

    def parse_csv(self):
        import pandas as pd
        import numpy as np
        self.df = pd.read_csv(self.filepath)
        print(self.df)





if __name__ == '__main__':
    # print_hi('PyCharm')
    ac = Action(filepath='./bvh/alpha_pose_input_xy_worldpos.csv')
