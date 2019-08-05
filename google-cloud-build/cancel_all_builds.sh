gcloud builds list | grep WORKING | cut -f 1 -d " " | xargs gcloud builds cancel
