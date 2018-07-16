#input_column: a,double,double,None,None,None
#input_type: SET

#output_column: b,double,double,None,None,None
#output_type: EMITS
library(forecast)
run <- function(ctx) {
   ctx$emit(0.0)
}
