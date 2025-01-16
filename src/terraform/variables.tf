#private pem key is set as TF_VAR_private_pem_key in .bashrc and points to the file "theObservatory.pem" 
variable "private_pem_key" {
  description = "Path to private key file"
  type        = string
  default     = "" 
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "Name of the S3 bucket for face recognition"
  type        = string
  default = "theobservatory"
}