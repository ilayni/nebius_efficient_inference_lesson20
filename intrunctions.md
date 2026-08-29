The assignment is now up on Google Classroom as well. Deadline: June 27.

You can complete it on any GPU - no need to use L40s or H100s specifically. Colab or Kaggle notebooks work fine too.

If you run into issues getting GPU access on Colab/Kaggle, reach out and I'll arrange a slot for you.

I ran this notebook in Colab on A100. You can find the specs here https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
To clarify the actual GPU memory size you can run 
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

But I had to 'break the law' and edit the cell with the specs