apt list | grep installed | cut -f 1,2 -d " " | sed "s#/now #,#g" | sort -f -d 
