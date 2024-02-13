# Lecture 2 AMATH 352 

* Euclidean norm

  Allows us to compare the size of two vectors: if $||x_1||_2 > ||x_2||_2 $ then $x_1$ is longer than $x_2$

* The 1-norm of a vector $||\underline x||_1 := |x_1| + |x_2|+...+|x_d| =\sum x_j$

  Given $\underline x$ we define its $\infin-norm$ as $||\underline x||_\infin := max(j=1,2,...,d)|x_j|$

  Verification that the 1-norm is indeed a norm:

  1. (non-negativeness) $||\underline x||_1=\sum|x_j|\ge0$ since each $|x_j|\ge0$

  2. (definiteness) suppose $||\underline x||_1 = 0$ and $x_k \ne 0 $ then $|x_k|>0$ and so $\sum |x_j|>0$ and therefore is a contradiction
  3. (post homogeneity) $||\alpha\cdot\underline x||_1 = \sum |\alpha x_j| = \sum |\alpha|\cdot|x_j| = |\alpha|*\sum=|\alpha|\cdot*||\underline x||_1$ 
  4. (triangle inequality) We already know that for any real a, b that |a+b| <= |a|+|b|. Then $||\underline x + \underline y|| <\sum ||\underline x_j|| + ||\underline y_j||\le\sum |x_j|+|y_j| = \sum|x_j|+\sum|y_j|=||\underline x||_1+ ||\underline y||_1$ 

* We say that a function from $R^d$ to $R$ is a norm, Notation $||\underline x||$ for $\underline x$ of any $R^d$  if it satisfies the following four axioms:

  1. $||\underline x||\ge0$ for any x belongs to R^d (non negative)
  2. it equals 0 if any only if x = $\empty$
  3. if $a\in R$ is any scalar then $||\alpha\cdot\underline x||= |\alpha|\cdot||\underline x||$
  4. $||\underline x + \underline y|| < ||\underline x|| + ||\underline y||$ (triangle inequality)

  