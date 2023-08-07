function delete_confirm(delete_file_id, delete_file_name){
    $("#delete_file_id").val(delete_file_id)
    $("#delete_file_name").text(delete_file_name)
}

$(document).ready(function(){ 
  $('#upload_file').on('change', function(){
    var fileName = $(this).val().replace("C:\\fakepath\\", "");

    $(this).next('.custom-file-label').html(fileName);
  });

  $('#data_source_lst').DataTable();
});