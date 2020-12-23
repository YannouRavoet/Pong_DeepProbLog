# PONG DeepProbLog
Made in context of the Capita Selecta Course: AI - Neurosymbolic approaches (2020). 
This program runs an ai opponent for the game of Pong. It perceives the screen through 
a CNN and reasons on which action to take with first-order logic.
 
## TASKS
1. Implement Pong Game in Python 
    * 28x28 screen
        - ball size = 1 pixel
        - paddles = 1x3 pixels
    * Human Player vs AI Player (receives screen each update)
    * AI logic model (.pl file)
2. Generate Raw Data
    * Contains positions of player, opponent and ball
    * Divide into train and test dataset
3. Compute First logic data in the form of queries
    * Based on position of the ball and player, decide if action is noop, up or down
    * X = img_id, Y=desired_action
        - query = choose_action(train(img_id), desired_action)
        - query = choose_action(test(img_id), desired_action)
4. Train ai CNN model with queries
5. Write python part of ai model
    * player position is set at start and kept track of
    * player position is passed when querrying logic program for desired_action
    * opponent position is not taken into account
    

