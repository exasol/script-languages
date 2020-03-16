apt list | grep installed | cut -f 1,2 -d " " | sed "s#/.* #|#g" | sort -f -d 
