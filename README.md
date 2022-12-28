# Neural Abacus

Train a neural network to perform basic arithmetic on an Abacus (more specifically a Soroban). Currently capable of addition, subtraction, and representing numbers.

A modified GPT architecture generates a sequence of actions to manipulate the beads in different columns of the abacus. The model typically reaches 98% accuracy within 10 epochs.

GPT implementation by [MinGPT](https://github.com/karpathy/minGPT).

## Short Example
Here's the model's solution to 23 - 17:
```
_______________________________________
o  o  o  o  o  o  o  o  o  o  o  o  o
                                     
---------------------------------------
                        o  o         
o  o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o  o  o  o  o  o  o     o  o  o
o  o  o  o  o  o  o  o  o  o  o  o  o
=======================================
0  0  0  0  0  0  0  0  2  3  0  0  0

Step 0. Remove 1 bottom bead(s) in column 8
_______________________________________
o  o  o  o  o  o  o  o  o  o  o  o  o
                                     
---------------------------------------
                        o  o         
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o     o  o  o
o  o  o  o  o  o  o  o  o  o  o  o  o
=======================================
0  0  0  0  0  0  0  0  1  3  0  0  0

Step 1. Add top bead in column 9
_______________________________________
o  o  o  o  o  o  o  o  o     o  o  o
                           o         
---------------------------------------
                        o  o         
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o     o  o  o
o  o  o  o  o  o  o  o  o  o  o  o  o
=======================================
0  0  0  0  0  0  0  0  1  8  0  0  0

Step 2. Remove 2 bottom bead(s) in column 9
_______________________________________
o  o  o  o  o  o  o  o  o     o  o  o
                           o         
---------------------------------------
                        o  o         
o  o  o  o  o  o  o  o        o  o  o
o  o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o  o
=======================================
0  0  0  0  0  0  0  0  1  6  0  0  0

Step 3. Remove 1 bottom bead(s) in column 8
_______________________________________
o  o  o  o  o  o  o  o  o     o  o  o
                           o         
---------------------------------------
                           o         
o  o  o  o  o  o  o  o  o     o  o  o
o  o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o  o
=======================================
0  0  0  0  0  0  0  0  0  6  0  0  0

Step 4. STOP
Correct: 23.000 - 17.000 = 6.000
```

## Long Example
98478.518 + 113052554.658
```
_______________________________________
o  o  o  o  o        o           o   
               o  o     o  o  o     o
---------------------------------------
               o  o  o  o  o     o  o
o  o  o  o  o  o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o  o  o  o     o  o     o  o   
o  o  o  o  o     o     o  o  o  o  o
=======================================
0  0  0  0  0  9  8  4  7  8  5  1  8

Step 0. Add 1 bottom bead(s) in column 1
_______________________________________
o  o  o  o  o        o           o   
               o  o     o  o  o     o
---------------------------------------
   o           o  o  o  o  o     o  o
o     o  o  o  o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o  o  o  o     o  o     o  o   
o  o  o  o  o     o     o  o  o  o  o
=======================================
0  1  0  0  0  9  8  4  7  8  5  1  8

Step 1. Add 1 bottom bead(s) in column 2
_______________________________________
o  o  o  o  o        o           o   
               o  o     o  o  o     o
---------------------------------------
   o  o        o  o  o  o  o     o  o
o        o  o  o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o  o  o  o     o  o     o  o   
o  o  o  o  o     o     o  o  o  o  o
=======================================
0  1  1  0  0  9  8  4  7  8  5  1  8

Step 2. Add 3 bottom bead(s) in column 3
_______________________________________
o  o  o  o  o        o           o   
               o  o     o  o  o     o
---------------------------------------
   o  o  o     o  o  o  o  o     o  o
o        o  o  o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o     o  o     o  o     o  o   
o  o  o  o  o     o     o  o  o  o  o
=======================================
0  1  1  3  0  9  8  4  7  8  5  1  8

Step 3. Add top bead in column 5
_______________________________________
o  o  o  o  o  o     o           o   
                  o     o  o  o     o
---------------------------------------
   o  o  o     o  o  o  o  o     o  o
o        o  o  o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o     o  o     o  o     o  o   
o  o  o  o  o     o     o  o  o  o  o
=======================================
0  1  1  3  0  4  8  4  7  8  5  1  8

Step 4. Add 1 bottom bead(s) in column 4
_______________________________________
o  o  o  o  o  o     o           o   
                  o     o  o  o     o
---------------------------------------
   o  o  o  o  o  o  o  o  o     o  o
o        o     o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o     o  o     o  o     o  o   
o  o  o  o  o     o     o  o  o  o  o
=======================================
0  1  1  3  1  4  8  4  7  8  5  1  8

Step 5. Add top bead in column 6
_______________________________________
o  o  o  o  o  o  o  o           o   
                        o  o  o     o
---------------------------------------
   o  o  o  o  o  o  o  o  o     o  o
o        o     o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o     o  o     o  o     o  o   
o  o  o  o  o     o     o  o  o  o  o
=======================================
0  1  1  3  1  4  3  4  7  8  5  1  8

Step 6. Remove 3 bottom bead(s) in column 6
_______________________________________
o  o  o  o  o  o  o  o           o   
                        o  o  o     o
---------------------------------------
   o  o  o  o  o     o  o  o     o  o
o        o     o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o     o  o  o  o  o     o  o   
o  o  o  o  o     o     o  o  o  o  o
=======================================
0  1  1  3  1  4  0  4  7  8  5  1  8

Step 7. Add top bead in column 5
_______________________________________
o  o  o  o  o     o  o           o   
               o        o  o  o     o
---------------------------------------
   o  o  o  o  o     o  o  o     o  o
o        o     o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o     o  o  o  o  o     o  o   
o  o  o  o  o     o     o  o  o  o  o
=======================================
0  1  1  3  1  9  0  4  7  8  5  1  8

Step 8. Remove 4 bottom bead(s) in column 5
_______________________________________
o  o  o  o  o     o  o           o   
               o        o  o  o     o
---------------------------------------
   o  o  o  o        o  o  o     o  o
o        o     o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o     o  o  o  o  o     o  o   
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  4  7  8  5  1  8

Step 9. Add top bead in column 7
_______________________________________
o  o  o  o  o     o              o   
               o     o  o  o  o     o
---------------------------------------
   o  o  o  o        o  o  o     o  o
o        o     o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o     o  o  o  o  o     o  o   
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  9  7  8  5  1  8

Step 10. Add top bead in column 8
_______________________________________
o  o  o  o  o     o     o        o   
               o     o     o  o     o
---------------------------------------
   o  o  o  o        o  o  o     o  o
o        o     o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o     o  o  o  o  o     o  o   
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  9  2  8  5  1  8

Step 11. Add 1 bottom bead(s) in column 7
_______________________________________
o  o  o  o  o     o     o        o   
               o     o     o  o     o
---------------------------------------
   o  o  o  o        o  o  o     o  o
o        o     o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o     o  o  o  o  o     o  o   
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  9  2  8  5  1  8

Step 12. Add top bead in column 9
_______________________________________
o  o  o  o  o     o     o  o     o   
               o     o        o     o
---------------------------------------
   o  o  o  o        o  o  o     o  o
o        o     o  o  o  o  o  o     o
o  o  o  o  o  o  o  o     o  o  o  o
o  o  o     o  o  o  o  o     o  o   
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  9  2  3  5  1  8

Step 13. Remove 1 bottom bead(s) in column 9
_______________________________________
o  o  o  o  o     o     o  o     o   
               o     o        o     o
---------------------------------------
   o  o  o  o        o  o  o     o  o
o        o     o  o  o  o  o  o     o
o  o  o  o  o  o  o  o        o  o  o
o  o  o     o  o  o  o  o  o  o  o   
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  9  2  2  5  1  8

Step 14. Add 1 bottom bead(s) in column 8
_______________________________________
o  o  o  o  o     o     o  o     o   
               o     o        o     o
---------------------------------------
   o  o  o  o        o  o  o     o  o
o        o     o  o  o  o  o  o     o
o  o  o  o  o  o  o  o  o     o  o  o
o  o  o     o  o  o  o     o  o  o   
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  9  3  2  5  1  8

Step 15. Add top bead in column 10
_______________________________________
o  o  o  o  o     o     o  o  o  o   
               o     o              o
---------------------------------------
   o  o  o  o        o  o  o     o  o
o        o     o  o  o  o  o  o     o
o  o  o  o  o  o  o  o  o     o  o  o
o  o  o     o  o  o  o     o  o  o   
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  9  3  2  0  1  8

Step 16. Add 1 bottom bead(s) in column 10
_______________________________________
o  o  o  o  o     o     o  o  o  o   
               o     o              o
---------------------------------------
   o  o  o  o        o  o  o  o  o  o
o        o     o  o  o  o  o        o
o  o  o  o  o  o  o  o  o     o  o  o
o  o  o     o  o  o  o     o  o  o   
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  9  3  2  1  1  8

Step 17. Add 1 bottom bead(s) in column 9
_______________________________________
o  o  o  o  o     o     o  o  o  o   
               o     o              o
---------------------------------------
   o  o  o  o        o  o  o  o  o  o
o        o     o  o  o  o  o        o
o  o  o  o  o  o  o  o  o  o  o  o  o
o  o  o     o  o  o  o        o  o   
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  9  3  3  1  1  8

Step 18. Add top bead in column 11
_______________________________________
o  o  o  o  o     o     o  o  o      
               o     o           o  o
---------------------------------------
   o  o  o  o        o  o  o  o  o  o
o        o     o  o  o  o  o        o
o  o  o  o  o  o  o  o  o  o  o  o  o
o  o  o     o  o  o  o        o  o   
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  9  3  3  1  6  8

Step 19. Remove 2 bottom bead(s) in column 12
_______________________________________
o  o  o  o  o     o     o  o  o      
               o     o           o  o
---------------------------------------
   o  o  o  o        o  o  o  o  o  o
o        o     o  o  o  o  o         
o  o  o  o  o  o  o  o  o  o  o  o  o
o  o  o     o  o  o  o        o  o  o
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  9  3  3  1  6  6

Step 20. Add 1 bottom bead(s) in column 11
_______________________________________
o  o  o  o  o     o     o  o  o      
               o     o           o  o
---------------------------------------
   o  o  o  o        o  o  o  o  o  o
o        o     o  o  o  o  o     o   
o  o  o  o  o  o  o  o  o  o  o     o
o  o  o     o  o  o  o        o  o  o
o  o  o  o  o  o  o     o  o  o  o  o
=======================================
0  1  1  3  1  5  0  9  3  3  1  7  6

Step 21. STOP
Correct: 98478.518 + 113052554.658 = 113151033.176
```