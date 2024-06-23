We have found that for 2 by 2 confusion matrices (a common summary relating the relation between categorical variables) the expected value of the [xicor](https://arxiv.org/abs/1909.10140) coefficient of correlation specializes into the re-normalized square of the determinant!

One can summarize how a 0/1 variable x relates to a 0/1 variable y as by writing down:


  * The true positives (`tp`), the number of times `x = 1` and `y = 1`.
  * The false positives (`fp`), the number of times `x = 1` and `y = 0`.
  * The true negatives (`tn`), the number of times `x = 0` and `y = 0`.
  * The false negatives (`fn`), the number of times `x = 0` and `y = 1`.


These four numbers can be organized in convenient table called "the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)" as follows.

<table>
<tr><td></td><td><b>x = 0</b></td><td><b>x = 1</b></td></tr>
<tr><td><b>y = 1</b></td><td>fn</td><td>tp</td></tr>
<tr><td><b>y = 0</b></td><td>tn</td><td>fp</td></tr>
</table>

The [xicor](https://arxiv.org/abs/1909.10140) coefficient is itself a random variable over permutations of the items that are "x ties" (have the same x value). The individual draws of the xicor estimate can be wild (and even include negative values). However, the expected value can be estimated with the following determinant formula (our, presumably new, result):

$$\frac{ \begin{array}{|cc|} fn & tp \\ tn & fp \\ \end{array}^2 }{\left(fn + tn\right) \cdot \left(fn + tp\right) \cdot \left(fp + tn\right) \cdot \left(fp + tp\right)}$$

The above is, in my opinion, quite beautiful. It allows confirmation of a number of known properties of the expected value of xicor (in this case) to be read of quickly (such as symmetries and the non-negativity of the expected value).

The derivation of this, and some consequences, can be found [here](https://github.com/WinVector/Examples/blob/main/XICOR/xicor_indicator_classification.ipynb).
