# S3 bucket
resource "aws_s3_bucket_versioning" "images_versioning" {
  bucket = aws_s3_bucket.the-observatory-bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket for known faces
resource "aws_s3_bucket" "the-observatory-bucket" {
  bucket = "the-observatory-bucket"
    lifecycle {
    prevent_destroy = false
  }
}

# # Upload an known face to the S3 known faces Bucket
# resource "aws_s3_object" "image1" {
#   bucket       = aws_s3_bucket.the-observatory-known-faces.id
#   key          = "PatrickStewart.jpg"    # Key for the object in the bucket
#   source       = "./face_upload/known_faces/PatrickStewart.jpg" # Local path to the image
#   content_type = "image/jpeg"             # MIME type
# }
# # Upload an unknown face to the S3 faces to check Bucket
# resource "aws_s3_object" "image2" {
#   bucket       = aws_s3_bucket.the-observatory-faces-to-check.id
#   key          = "PatrickStewart2004.jpg"    # Key for the object in the bucket
#   source       = "./face_upload/faces_to_check/PatrickStewart2004-08-03.jpg" # Local path to the image
#   content_type = "image/jpeg"             # MIME type
# }
# # Upload an unknown face to the S3 faces to check Bucket
# resource "aws_s3_object" "image3" {
#   bucket       = aws_s3_bucket.the-observatory-faces-to-check.id
#   key          = "pexels-olly.jpg"    # Key for the object in the bucket
#   source       = "./face_upload/faces_to_check/pexels-olly-712513.jpg" # Local path to the image
#   content_type = "image/jpeg"             # MIME type
# }