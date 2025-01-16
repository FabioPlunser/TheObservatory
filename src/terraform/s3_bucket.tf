resource "aws_s3_bucket" "face_recognition_bucket" {
  bucket = var.bucket_name
  force_destroy = true  # Allows terraform to delete the bucket even if it contains objects
}

resource "aws_s3_bucket_cors_configuration" "face_recognition_bucket_cors" {
  bucket = aws_s3_bucket.face_recognition_bucket.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "DELETE"]
    allowed_origins = ["*"]  
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}