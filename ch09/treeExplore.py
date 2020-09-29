from tkinter import *

from numpy import *
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ch09 import regTrees

matplotlib.use('TkAgg')


def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = regTrees.create_tree(
            reDraw.rawDat, regTrees.model_leaf, regTrees.model_err, (tolS, tolN)
        )
        yHat = regTrees.create_fore_cast(myTree, reDraw.testDat, regTrees.model_tree_eval)
    else:
        myTree = regTrees.create_tree(
            reDraw.rawDat, ops=(tolS, tolN)
        )
        yHat = regTrees.create_fore_cast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0], reDraw.rawDat[:, 1], s=5)
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    reDraw.canvas.draw()


def get_inputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, "10")
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, "1.0")
    return tolN, tolS


def draw_new_tree():
    tolN, tolS = get_inputs()
    reDraw(tolS, tolN)


if __name__ == '__main__':
    root = Tk()
    reDraw.f = Figure(figsize=(5, 4), dpi=100)
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
    reDraw.canvas.draw()
    reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
    Label(root, text="tolN").grid(row=1, column=0)
    tolNentry = Entry(root)
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0, '10')
    Label(root, text="tolS").grid(row=2, column=0)
    tolSentry = Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0, '1.0')
    Button(root, text="Redraw", command=draw_new_tree).grid(
        row=1, column=2, rowspan=3
    )
    chkBtnVar = IntVar()
    chkBtn = Checkbutton(root, text="Model Tree", variable=chkBtnVar)
    chkBtn.grid(row=3, column=0, columnspan=2)
    reDraw.rawDat = array(regTrees.load_data_set("sine.txt"))
    reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
    reDraw(1.0, 10)
    root.mainloop()
