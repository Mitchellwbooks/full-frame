# full-frame
An Automated Image Quality and Label Tagger for Photography Workflows


## Goals
Photographers maintaining a large catalog of photographs often don't have time to rate all their photos; 
full-frame is designed to take some burden off the tagging and rating of pictures. 


## Definitions
- Tagging
  - A label or categories for a picture, stored in metadata.
  - Eg. A photo of a Tiger may have labels: "Tiger", "Animal"
- Rating
  - A qualitative assessment for a picture, stored in metadata.
  - EG. A photo of a Tiger with subject in focus, good depth of field, and well framed: "5 Stars"

## Project Goals
1. Automated tagging and rating of pictures
2. Editing Platform Agnostic, by use of common XML metadata file format.
   - Capture One
   - Photoshop
   - Etc
3. File System Portable
   - Allow re-installation of tool to any platform without need for maintaining central database.
4. User Informed
   - User's tags and ratings are trained on to inform the system 