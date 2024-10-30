import numpy as np
import copy


def matrice_to_string(grille):
    out = ""
    for ligne in grille:
        for element in ligne :
            out +=str(element)
    return out



def norm(a):    
    return (a/9)-.5


def denorm(a):
    return (a+.5)*9




def correspondance(x, y):
    cpt = np.sum(x == y)
    percentage = (cpt / np.prod(x.shape)) * 100
    print("percentage de correspondance :",str(percentage),"%")



def solver(grid,SudokuCNN):

    aux = copy.copy(grid) # pour ne pas modifier l'originale
    
    while(1):
    
        out = SudokuCNN.predict(aux.reshape((1,9,9,1)))  
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9,9))+1 
        prob = np.around(np.max(out, axis=1).reshape((9,9)), 2) 
        
        aux = denorm(aux).reshape((9,9))
        mask = (aux==0)
     
        if(mask.sum()==0):
            break
            
        prob2 = prob*mask
    
        ind = np.argmax(prob2)
        x, y = (ind//9), (ind%9)

        val = pred[x][y]
        aux[x][y] = val
        aux = norm(aux)
    
    return pred



# doit transformer une matrice 9x9 en un string de 81 caractére puis le transforme en array 9,9,1 puis le normalize puis le donne au solver
def solve_sudoku(game,SudokuCNN):
    
    game = matrice_to_string(game)
    game = np.array([int(j) for j in game]).reshape((9,9,1))
    game = norm(game)
    game = solver(game,SudokuCNN)
    return game



