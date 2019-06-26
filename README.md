# mlinseconds_ovelychko

Hello!

04/06/2019 Roman Koshlyak (here and after - RK) announced his initiative to teach 3 people basics of deep ML. 

https://www.facebook.com/groups/udsclub/2367546239958274

https://www.facebook.com/groups/195065914629311/

https://github.com/dev-rkoshlyak/mlinseconds

RK: Looking for people for new batch.
    We will start once we get around 30 people committed. We will be dropping people who suddenly realize that they have no time.
    Req:
    python knowledge
    5-15 hours per week of study time
    Comment if interested
    Please read in first comment feedback from people who will graduate soon to set up your expectations
      
RK: Details for the program
    Program will be 6-9 weeks long. Every week we will be solving one toy problem. 
    The idea behind each problem is to practice one or more machine learning technics. 
    At the end of every week we will drop some people who does not have time/very interested by now is super busy/etc
    Once we will be done with a toy problems. We will have 3 people left and we will do kaggle competition for one month. 
    With idea to apply deep machine learning to the problem.

--------------------------------------------------------------------------------------

1 week (till 09/06/2019) - HelloXOR

2 week (till 16/06/2019) - HelloXOR in less then 10 steps
https://www.facebook.com/notes/machine-learning-in-seconds-deep-learning-artificial-intelligence-ai/hello-xor/198583254277577/

RK: At this point, please share your git commit with solution for hello xor. 
    Also write down your work log. Write down what you tried, what worked and what did not work. 
    How much time you spend on what and what was improvement from it. Write down your pain points and how you overcome them

3 week (till 23/06/2019) - General CPU
https://www.facebook.com/notes/machine-learning-in-seconds-deep-learning-artificial-intelligence-ai/general-cpu/198587934277109/

    https://github.com/DeVyacheslav/Deep-learning-practice/blob/master/mlis/problems/generalCpu.py
    https://github.com/imylyanyk/mlinseconds/commit/985839414b3f51ee39043e8ceb390fa9b36a73d3
    https://github.com/panda4us/mlinseconds/blob/master/mlis/problems/generalCpu_Adam.py
    And actually setting up betas(mean and std) quite apart from the default ones in Adam allowed to go below the 10 steps on average
    https://github.com/vlyubin/mlinseconds/commit/84ec38c060c3bc0fb9f3e075bde04a70f6e40e27
    https://github.com/mpylypovych/mlinseconds/blob/master/mlis/problems/generalCpu.py
    https://github.com/mskoryk/mlinseconds_mskoryk/blob/master/generalCpu.py

RK: https://www.youtube.com/watch?v=5qefnAek8OA   
    that's stuff works okay if you have a lot of steps, it's broken if we have small number of steps. 
    Additionally, it add small "noise" to results of test data. It does not matter in cases when test data different from train data

RK: weight decay is useless for general cpu, because our train data = weight data. 
    Same about data permutation - should have no effect at all since we train on all data
RK: grid search on number of layers

4 week (till 30/06/2019) - findMe
https://github.com/dev-rkoshlyak/mlinseconds/blob/master/mlis/problems/findMe.py
RK: feel free to use handcrafted features in findMe, it obviously would helped with previous task, if you would make model linearly separable, but for findMe it would not be an option because of size.

