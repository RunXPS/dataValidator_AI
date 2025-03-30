# dataValidator_AI

### Getting Started
To reduce confusion about versions and differences in software, I've created a Docker dev-container with the necessary python packages.    
Starting the container may take a while (+5 minutes for my first time) however the next time should be much quicker.   
   
Dowload Docker **[here](https://www.docker.com/products/docker-desktop/)** if needed
![image](https://github.com/user-attachments/assets/80ad46ca-9ee4-4fc5-a526-755fe25ca540)


To adjust any of the packages you want to use in the project, modify the [requirements.txt](https://github.com/RunXPS/dataValidator_AI/blob/main/requirements.txt)![Uploading image.png…]()
 file. Some packages may already be commented out.   


### Meeting 03/30
- Model is searching for:
    1) **Industry**
    2) **Major**
    3) Hometown* 
    4) "Software" or "Hardware"*
    5) Time series ==> other datasets
        * Have the start dates, not end (@ company)
        * Years at company
        * **No EDU data in 2024**
    6) Industry 'alignment' ==> Did founder stay in the anticipated industry from their major (eg. CS ==> tech startup)
* Potentially