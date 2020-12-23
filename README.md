##PLANNING
1. Implement Pong Game in Python 
    * 28x28 screen
        - ball size = 1 pixel
        - paddles = 1x3 pixels
    * Human Player vs AI Player (receives screen each update)
2. Generate Raw Data
    * Contains positions of player, opponent and ball
3. Compute First logic data
    * Based on position of the ball and player, decide if action is noop, up or down
    * X = img_id, Y=desired_action
4. Write First ProbLog logic
    * NN => input = (1,28,28) (same as MNIST example)
            output = (x,y) of ball
    * player position is given at start and kept track of
    * player action is based on whether y-coord of ball is <,= or > than y-coord of player
    * opponent position is not taken into account
    

