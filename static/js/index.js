$(function () {
  $("button.selector-button").on("click", function () {
    $("button.selector-button").removeClass("selected");
    $(this).addClass("selected");
  });
  $("button#videos-tool-button").trigger("click");

  $("button#upload-video-button").on("click", function () {
    const uploadForm = $("<form/>")
      .attr("action", "/api/video/upload")
      .attr("method", "post")
      .attr("enctype", "multipart/form-data")
      .attr("target", "formdummy");
    $("<h2/>")
      .text("Upload video")
      .addClass("red-hat-bold")
      .css("margin-top", "0.5em")
      .appendTo(uploadForm);
    const p1 = $("<p/>");
    $("<label/>")
      .text("File: ")
      .append(
        $("<input/>")
          .attr("name", "file")
          .attr("type", "file")
          .prop("required", true)
      )
      .appendTo(p1);
    p1.appendTo(uploadForm);
    const p2 = $("<p/>");
    $("<label/>")
      .attr("for", "name")
      .text("Description:")
      .append($("<br/>"))
      .append($("<input/>").attr("name", "name").attr("type", "text"))
      .appendTo(p2);
    p2.appendTo(uploadForm);
    $("<input/>")
      .attr("type", "submit")
      .attr("value", "Upload")
      .appendTo(uploadForm);
    uploadForm.appendTo("#popup");
    $(".popups").show();
  });
});
