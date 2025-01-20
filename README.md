# Virtual Clothing Try-On Web App

This is a web app that allows users to upload an image of their chosen clothing (with the background removed) and a video of themselves (one person, facing the camera) to virtually try on the clothing. After uploading the necessary files and selecting the desired clothing, users can generate a video where the selected clothing is applied to their image.

### Key Features:
- **Upload Clothing Image**: Users can upload an image of clothing with the background removed (e.g., PNG format).
- **Upload Personal Video**: Users upload a video of themselves facing the camera.
- **Virtual Try-On**: The app combines the uploaded video and clothing image to generate a video of the clothing virtually fitting the user.
- **Download Processed Video**: After processing, users are notified when the video is ready for download, with a link to download the final video.

## Prerequisites

To run this project locally, you will need to set up a virtual environment and install the required dependencies. Here's how to do it:

**Before cloning the repository, change your working directory to the folder where you want the project to be saved:**

Navigate to the directory where you want to store the project:
```bash
cd /path/to/your/directory
```

### 1. Clone the repository:
```bash
git clone https://github.com/johnnietse/cloth-virtual-try-on.git
cd cloth-virtual-try-on
```

### 2. Set up a virtual environment:
If you're using Python 3, you can set up a virtual environment with the following commands:
```bash
python3 -m venv venv
```

### 3. Activate the virtual environment:
- For Windows:
```bash
venv\Scripts\activate
```

- For macOS/Linux:
```bash
source venv/bin/activate
```

### 4. Install dependencies:
Once your virtual environment is activated, install the required dependencies using **requirements.txt**:
```bash
pip install -r requirements.txt
```

## Running the App Locally
### 1. Start the Flask server:
After setting up the environment and installing dependencies, you can run the web app using the following command:
```bash
flask run
```

This will start a local development server, and you should see an output like this in the terminal:
```bash
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

### 2. Access the Web App:
Once the server is running, open your web browser and go to the following URL:
```bash
http://127.0.0.1:5000
```
You should see the web app interface where you can upload your clothing image and personal video.

### 3. Upload Files and Process:
- Upload your **clothing image** (make sure the background is removed).
- Upload your **personal video** (where you are facing the camera).
- Select the clothing item you want to try on.
- Click the **"Upload and Process"** button to generate the virtual try-on video.
  
### 4. Download the Processed Video:
Once the video has been processed, you will see a message:

```bash
Video processing complete! Click the link below to download.
```

Click on the **"Download Processed Video"** button to download the generated video.

## Sample Files
To get an idea of how the app works, you can look at the following sample files within the repository:

- Clothing Images: Sample clothing images can be found in the **Resources/Shirts** folder.
- Uploaded Videos: Sample user videos can be found in the **static/uploads** folder.
- Processed Videos: Sample processed virtual clothing try-on videos are available in the **static/processed** folder.

These samples demonstrate the input and output for the app's functionality.

## Future Considerations
- Deployment to a cloud platform (e.g., Render, Heroku, AWS).
- Enhancements to virtual clothing fitting (e.g., improved image processing and fit accuracy).
- Support for multiple clothing items or different video orientations.
