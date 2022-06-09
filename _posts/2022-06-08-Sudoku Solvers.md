---
layout: post
title:  "Using deep learning to help solve Sudoku puzzles"
categories: blog assignment
permalink: posts/blog-post-final
author: Gursimrit, Roman, Nicola
---

In this blog post we'll talk about how we used deep learning to create a Sudoku solver. 
The link to our Github repo can be found here: https://github.com/romakouz/sudoku.project

In the repository there is also a helpful README file that you guys can check out.

## Project Overview
Sudoku is a game enjoyed by millions of people and we wanted to create a web app where people could upload images of their sudoku puzzles and have them be solved.

First we worked on obtaining a dataset containing Sudoku puzzles and their digits. We ended up finding a dataset that contained a million Sudoku puzzles along with the answers to each of the puzzles. 
We then uploaded this dataset into our Google Colab. The data was originally just a string of 81 digits, where 0's represented empty cells, so we had to then format it into the standard 9x9 Sudoku grid.

Next we made a backtracking algorithm that could solve these Sudoku puzzles with a 100% success rate as long as the given puzzle was valid in it of itself. 

Then we found a dataset of 200 Sudoku images which were used to train an image classifying model. However, before we could do that we
needed to pre-process the images by grayscaling, blurring, reshaping, and etc so they could all be uniform. Once that was done we split
each Sudoku puzzle into their individual cells. We then used those cells to train our Linear Regression model to classify those digits.
Afterwards we tested the LR model and obtained about 90% accuracy. We decided to then use a CNN model to classify the Sudoku cells. After
training, we obtained a 95% accuracy on our test data. Going forward, we decided to use the CNN model over the LR model.

Now we worked on creating our web app. First we pickled our CNN model so it could saved and didn't need to be retrained. We then
made our web app through flask. Then we created three .html files titled 'base', 'main_better', and 'submit' in our templates folder
on our Github page which helped to establish the general layout of the web page.

Within our web page we made a function where the user could correct one of the puzzle's entries if our model had any uncertainty regarding
classifying any digits. Along with another function this process is repeated until the user decides they are done correcting their
Sudoku puzzle.

In the end, our web page would ask users to submit their Sudoku puzzle image (any size), ask them if any corrections are needed, and then finally
give them a solution to their Sudoku puzzle.

![alt text](https://i.gyazo.com/3cab1370dbe42794c91338d4dbede146.png)

## Backtracking Algorithm

We decided to use a backtracking algorithm to solve our Sudoku puzzles because an algorithmic approach to solving the puzzles would guarantee a correct result a 100% of the time
as long as the given puzzle itself was valid in the first place. This way our given solution would always be correct and if we instead decided to use machine learning to solve the
puzzles then we wouldn't get 100% accuracy which isn't helpful at all for the user. Along with this, the backtracking algorithm was much quicker at producing a solution compared
to a machine learning approach. 

Backtracking is a form depth-first search. For each empty cell, the algorithm inserts a valid number and then checks the validity conditions of Sudoku are met with this number. If
the conditions are valid then move to a new cell, if they are invalid pick the next number up in the list. This process is then repeated until all values in the cell's list have 
failed the validity check or all cells have been filled. If the cell's values have been exhausted, the algorithm then moves to a previous cell and picks a different value and repeats 
the process over.

```python
#function to solve recursively using backtracking, using
def solve(puzzle):
  '''
  recursively solves puzzle until all cells are filled in (validly)
  using backtracking
  '''
  #find next empty cell, if none are empty, puzzle is solved
  find = find_cell(puzzle)
  if not find:
      return True
  else:
      #row, col store the location of the empty cell
      row, col = find

  for i in range(1,10):
      #check if given entry is valid at location row, col
      if valid(puzzle, i, (row, col)):
          #if valid, input given entry at location
          puzzle[row][col] = i
          #check if we can solve the puzzle with the entry we used
          if solve(puzzle):
              return puzzle, True
          #if we cannot, reset entry to 0, try again
          puzzle[row][col] = 0

  return False
```

![alt text](https://i.gyazo.com/fa67b6b19b4dd6157adc208832e07bee.png)

## CNN Model to Classify Images

A convolutional neural network (CNN) is a type of machine learning model that can classify images. It takes images as inputs, extracts, and leanrs the features of the image, and then classifies them
based on the learned features. We used a CNN model on the thousands of Sudoku cells we created when we split them up individually from our dataset containing 200 Sudoku puzzle images. From training our CNN 
model we were able to obtain a 95% accuracy on our test data.

Our CNN model contains multiple convolutional layers, max-pooling, dropout, and dense layers. We alternated between Conv2D and MaxPooling 2D layers. The dropout layers then provide some randomness to the model.
Flatten makes the image go from 2D to 1D since our dense layer requres images to be 1D.

```python
model1 = models.Sequential([
      tf.keras.Input(shape=(50, 50, 1)),
      layers.Conv2D(filters = 32, kernel_size = (3,3), data_format = 'channels_last', activation = 'relu', input_shape = (50,50,1)),
      layers.MaxPooling2D(pool_size = (2,2)),
      layers.Dropout(0.2),
      layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'),
      layers.MaxPooling2D(pool_size = (2,2)),
      layers.Dropout(0.2),
      layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
      layers.MaxPooling2D(pool_size = (2,2)),
      layers.Dropout(0.1),
      layers.Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'),
      layers.MaxPooling2D(pool_size = (2,2)),
      layers.Flatten(),
      layers.Dense(128, activation = 'relu'),
      layers.Dense(10)
])
```

![alt text](https://i.gyazo.com/1d9ef7a56fddbb32d384211f498d90b2.png)

## Correcting Function in our Web App

Due to the uncertainty of machine learning, no model can ever be 100% accurate. The correcting1 function, will ask the user if they would like to correct one of the puzzle's entries after the model had
uncertainity in its prediction. If the user clicks yes, then the function routes to submit where it will prompt the html code that allows the user to change an entry. Otherwise, it will solve the puzzle 
and render the submit file to display it. This function will never be called by the user, but rather by the submit.html file after they have submitted a puzzle. 

Next is the correcting function, which is run by the submit template after a user has specified that they would like to correct an entry, and allows them to specifiy the location as a pair and a digit to place 
inside the sudoku puzzle. After correcting the puzzle, the function will display the newly corrected puzzle, then render the submit function to ask the user if they are ready to solve the function or if they 
would like to correct more entries. 

```python
@app.route('/correcting', methods=['POST'])
def correcting():
    #use global c_puzzle and copy it
    global c_puzzle
    global puzzle
    
    #get correction from user
    corr = request.form['correction']
    corr_tuple = tuple([int(i) for i in corr.split(',')])

    #correct puzzle using correct function from earlier
    c_puzzle = correct(corr_tuple,c_puzzle)
    c_puzzle_str = pf.print_puzzle(c_puzzle)

    puzzle_str = pf.print_puzzle(puzzle)


    return render_template('submit.html', adjusting=True, adjustment=corr, new_puzzle=c_puzzle_str, old_puzzle=puzzle_str)
```

![alt text](https://i.gyazo.com/1cf565fccb8254c5eab9e0742a780505.png)

## Conclusion

Throughout this project we learned more about machine learning and flask.

We used two machine learning models, Logistic Regression and CNN, in order to classify images which could then be put into a 9x9 array and be solved.
We used flask to create our web app that allows users to interact with it such as submitting images and entering in new values to correct a Sudoku cell. 

We hope you enjoyed reading this blog post and can now replicate this Sudoku Solver using our Github link and perhaps even improve it further!


