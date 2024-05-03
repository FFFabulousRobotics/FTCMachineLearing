

$(function() {
  $("button.selector-button").on("click", function() {
    $("button.selector-button").removeClass("selected");
    $(this).addClass("selected");
  });
  $("button#videos-tool-button").trigger("click");
});