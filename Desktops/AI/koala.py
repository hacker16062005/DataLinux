import tkinter as tk
from PIL import ImageTk
from PIL import Image
import time
class SimpleApp(object):
    def __init__(self, master, filename, **kwargs):
        self.master = master
        self.filename = filename
        self.canvas = tk.Canvas(master, width=500, height=500)
        self.canvas.pack()
        self.image = Image.open(self.filename)
        self.angle = 0

        self.master.after(100, self.draw)
 
    def draw(self):
        while self.angle < 90:
            tkimage = ImageTk.PhotoImage(self.image.rotate(angle))
            canvas_obj = self.canvas.create_image(
                250, 250, image=tkimage)
            self.angle += 5
            self.master.after(100, self.draw)

              
root = tk.Tk()
app = SimpleApp(root, 'koala.png')
root.mainloop()
