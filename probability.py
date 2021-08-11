def single_adjusted_probability(mask1, mask2, dist):
    if dist>100:
        return 0
    else:
        prob = (100-dist)/100
        
        if mask1.item() == 1 & mask2.item() == 1:
            return 0.6*prob
        elif mask1.item() == 1 & mask2.item() == 0:
            return 0.8*prob
        elif mask1.item() == 0 & mask2.item() == 1:
            return 0.8*prob
        elif mask1.item() == 0 & mask2.item() == 0:
            return prob
        
def final_probability(prob_vect):
    length = 0
    for i in prob_vect:
        if i != 0:
            length = length+1
    
    if length==0:
        return 0
    else:
        return sum(prob_vect)/length
