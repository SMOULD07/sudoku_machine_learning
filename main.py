import numpy as np
from resolution import *
from reconnaisance2 import *
from extraction import *
import keras
import torch



# main function
def resultat(chemin,model_reconnaisance,model_resolution):
   
    grille,_,_ = get_sudoku_predictions(chemin,model_reconnaisance)
    grille=np.array(grille).reshape(9,9)

    # chargement du modele de résolution 
    predi = outils.solve_sudoku(grille,model_resolution)
    
    print(grille)
    print(predi)

# app = custoom.Application()
# app.mainloop()

# model1 = model.Net()
# model1.load_state_dict(torch.load('/home/etud/Bureau/projet_Sudoku/reconnaisance2/mnist_model.pth'))
# model1.eval()

# model2 = keras.models.load_model('/home/etud/Bureau/projet_Sudoku/resolution/sudoku_model.h5')

# resultat("images/sudoku.png",model1,model2)

